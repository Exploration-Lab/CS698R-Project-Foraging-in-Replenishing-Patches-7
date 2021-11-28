import gym
import gym_env

import numpy as np


env = gym.make("foraging-replenishing-patches-v0")

env.reset()
for _ in range(10):
    action = np.random.randint(9)
    state, reward, done, _ = env.step(action)
    print(action, state, reward, done)
    if done:
        break

