import torch.nn as nn
import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import (
    create_mlp,
)
from typing import Type
from torch.nn import functional as F


# cherry picked from
# https://raw.githubusercontent.com/ku2482/fqf-iqn-qrdqn.pytorch/11d70bb428e449fe5384654c05e4ab2c3bbdd4cd/fqf_iqn_qrdqn/network.py


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_cosines=64, embedding_dim=7 * 7 * 64):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = nn.Linear

        self.net = nn.Sequential(
            linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):

    def __init__(self, net_arch, embedding_dim=7 * 7 * 64, activation=nn.ReLU):
        super(QuantileNetwork, self).__init__()
        layers = create_mlp(embedding_dim, 1, net_arch, activation)
        self.net = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, s_a_embedding, tau_embeddings):
        assert s_a_embedding.shape[0] == tau_embeddings.shape[0]
        assert s_a_embedding.shape[1] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = s_a_embedding.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        s_a_embedding = s_a_embedding.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (s_a_embedding * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        quantiles = self.net(embeddings)
        return quantiles.view(batch_size, N, 1)


class FractionProposalNetwork(nn.Module):

    def __init__(self, N=32, embedding_dim=7 * 7 * 64):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, N)
        )

        self.N = N
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):
        batch_size = state_embeddings.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)

        tau_0 = torch.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N + 1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, tau_hats, entropies


class FQFContinuousCritic(BaseModel):

    def __init__(
            self,
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            embedding_dim=7 * 7 * 64,
            N=32,
            num_cosines=32,
            activation_fn=nn.ReLU,
            normalize_images=True,
            n_critics=1,
            share_features_extractor=True,
            target=False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        action_dim = get_action_dim(self.action_space)

        # Feature extractor of Q Network.
        embedding_net = create_mlp(features_dim + action_dim, embedding_dim, net_arch, activation_fn)
        self.embedding_net = nn.Sequential(*embedding_net)

        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim, )
        # Quantile network.
        self.quantile_net = QuantileNetwork(net_arch=net_arch, embedding_dim=embedding_dim)

        # Fraction proposal network.
        self.fraction_net = FractionProposalNetwork(
            N=N, embedding_dim=embedding_dim)

        self.N = N
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.target = target

    def calculate_state_embeddings(self, states, actions):
        with torch.no_grad():
            features = self.extract_features(states)
        q_value_input = torch.cat([features, actions], dim=1)

        return self.embedding_net(q_value_input)

    def calculate_fractions(self, states, actions):
        state_embeddings = self.calculate_state_embeddings(states, actions)
        taus, tau_hats, entropies = self.fraction_net(state_embeddings)

        return taus, tau_hats, entropies

    def calculate_quantiles(self, states, actions, taus):
        state_embeddings = self.calculate_state_embeddings(states, actions)
        tau_embeddings = self.cosine_net(taus)

        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states, actions):

        taus, tau_hats, _ = self.calculate_fractions(
            states,
            actions
        )

        quantile_hats = self.calculate_quantiles(states, actions, tau_hats)

        # Calculate expectations of value distribution.
        q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats) \
            .sum(dim=1)

        return q

    def forward(self, obs, actions):
        q_value = self.calculate_q(obs, actions)
        return tuple([q_value])

    def q1_forward(self, obs, actions):
        q_value = self.calculate_q(obs, actions)
        return tuple([q_value])

