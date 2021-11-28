#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install stable-baselines3')


# In[1]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[18]:


class ForagingReplenishingPatches(gym.Env):
    def __init__(self, block_type=1, manual_play=False):
        self.reset_flag = False
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(8)
        self.block_type = block_type
        self.HARVEST_ACTION_ID = 8

        if self.block_type == 1:
            self.rewards = np.asarray([0, 70, 70, 0, 70, 0, 70, 0])
        elif self.block_type == 2:
            self.rewards = np.asarray([0, 0, 70, 70, 0, 70, 0, 70])
        elif self.block_type == 3:
            self.rewards = np.asarray([70, 0, 0, 70, 70, 0, 70, 0])

        self.rewarding_sites = np.arange(8)[self.rewards > 0]
        self.current_state = 0
        self.time_elapsed = 1.307

        self.farmer_reward = 0
        self.init_env_variables()
        if manual_play:
            self.init_foraging_img()
            self.manual_play()

    def replenish_rewards(self):
        if self.block_type == 1:
            replenish_rates = np.asarray([0, 4, 4, 0, 4, 0, 4, 0])
        elif self.block_type == 2:
            replenish_rates = np.asarray([0, 0, 8, 2, 0, 5, 0, 8])
        elif self.block_type == 3:
            replenish_rates = np.asarray([2, 0, 0, 4, 8, 0, 16, 0])
        replenish_rates[self.current_state] = 0
        self.rewards += replenish_rates
        self.rewards = np.clip(self.rewards, 0, 200)

    def step(self, action):
        self.time_elapsed += self.time_dist[str(self.current_state) + "to" + str(action)]
        self.current_state = action
        if self.time_elapsed >= 300:
            self.reset_flag = True
            return (self.current_state, 0 , self.reset_flag, {})
        
        self.time_elapsed += 1
        reward_old = self.farmer_reward
        if self.current_state in self.rewarding_sites:
            self.replenish_rewards()
            self.farmer_reward += self.rewards[self.current_state] * 0.90
            self.rewards[self.current_state] = (self.rewards[self.current_state] * 0.9)

        if self.time_elapsed >= 300:
            self.reset_flag = True
        return (self.current_state, self.farmer_reward - reward_old, self.reset_flag, {})

    def reset(self):
        self.reset_flag = False
        if self.block_type == 1:
            self.rewards = np.asarray([0, 70, 70, 0, 70, 0, 70, 0])
        elif self.block_type == 2:
            self.rewards = np.asarray([0, 0, 70, 70, 0, 70, 0, 70])
        elif self.block_type == 3:
            self.rewards = np.asarray([70, 0, 0, 70, 70, 0, 70, 0])
        self.rewarding_sites = np.arange(8)[self.rewards > 0]
        self.current_state = 0
        self.time_elapsed = 2
        self.farmer_reward = 0
        return self.current_state

    def render(self, mode="human"):
        print("Current State:", self.current_state, "Current Total Reward:", self.farmer_reward)

    def close(self):
        cv2.destroyAllWindows()
        return None

    def init_env_variables(self, first_point_angle=0):
        a = 1 / (2 * np.sin(np.pi / 8))  # fix a (radius) for unit side octagon
        self.octagon_points = np.asarray(
            [
                (
                    a * np.sin(first_point_angle + n * np.pi / 4),
                    a * np.cos(first_point_angle + n * np.pi / 4),
                )
                for n in range(8)
            ]
        )
        self.time_dist = {}
        for i in range(8):
            for j in range(8):
                dist = np.linalg.norm(self.octagon_points[i] - self.octagon_points[j])
                self.time_dist.update({str(i) + "to" + str(j): dist})


# In[19]:


from stable_baselines3.common.env_checker import check_env
env = ForagingReplenishingPatches(block_type=3)
check_env(env, warn=True)


# In[ ]:


env.reset()
for i in range(300):
    action = np.random.randint(8)
    state, reward, done, _ = env.step(action)
    print(action, state, reward, done)
    if done:
        break


# In[25]:


from stable_baselines3 import PPO
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10**5)


# In[26]:


obs = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      break


# In[27]:


# save
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Sem 5/CS698')
get_ipython().system('mkdir saved_models')
get_ipython().run_line_magic('cd', 'saved_models')

model.save("PPOmlpPolicy")


# In[28]:


print(model.policy)


# In[29]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Sem 5/CS698')
get_ipython().system('mkdir ppo_forager_tensorboard')
get_ipython().system('ls')


# In[30]:


from stable_baselines3 import PPO
env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_forager_tensorboard/")
model.learn(total_timesteps=10**6)


# In[32]:


# save
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Sem 5/CS698')
get_ipython().system('mkdir saved_models')
get_ipython().run_line_magic('cd', 'saved_models')

model.save("PPOmlpPolicy1M")


# In[31]:


get_ipython().system('tensorboard --logdir ./ppo_forager_tensorboard/')

