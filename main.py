from fqf_sac import FQFSAC, FQFSACPolicy
import gym

from Navigation2d import NavigationEnvAcc
from Navigation2d.config import obs_set, goal_set


env = NavigationEnvAcc({"OBSTACLE_POSITIONS": obs_set[1], "Goal": goal_set[-1]})
model = FQFSAC(FQFSACPolicy, env, verbose=1)
model.learn(total_timesteps=300000, log_interval=4)
