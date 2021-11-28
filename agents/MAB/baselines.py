#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install --upgrade --editable gym-env')


# In[2]:


import gym
import gym_env
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

env = gym.make('foraging-replenishing-patches-v0')

env.reset()
for i in range(300):
    action = np.random.randint(9)
    state, reward, done, _ = env.step(action)
    print(action, state, reward, done)
    if done:
        break


# In[3]:


def pureGreedy(env,maxEpisodes, maxTime=300, optimistic=True):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  if optimistic == True:
    Q = np.ones(env.action_space.n-1)*10**4

  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    while env.time_elapsed < maxTime:
        a = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
        s, r, terminal, info = env.step(a)
        s, r, terminal, info = env.step(8)
        N[a] += 1
        Q[a] += (r-reward-Q[a])/N[a]
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[3]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureGreedy(env=genv, maxEpisodes=50, optimistic=True)
print(rs)


# In[4]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureGreedy(env=genv, maxEpisodes=50, optimistic=False)
print(rs)


# In[4]:


def pureExplore(env,maxEpisodes, maxTime=300):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    while env.time_elapsed < maxTime:
        a = np.random.randint(env.action_space.n-1)
        s, r, terminal, info = env.step(a)
        s, r, terminal, info = env.step(8)
        N[a] += 1
        Q[a] += (r-reward-Q[a])/N[a]
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[6]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureExplore(env=genv, maxEpisodes=50)
print(rs)


# In[5]:


def epsGreedy(env,eps, maxEpisodes,maxTime=300):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    while env.time_elapsed < maxTime:
        if np.random.rand(1) < eps:
            a = np.random.randint(env.action_space.n-1)
        else:
            a = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))        
        s, r, terminal, info = env.step(a)
        s, r, terminal, info = env.step(8)
        N[a] += 1
        Q[a] += (r-reward-Q[a])/N[a]
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[8]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = epsGreedy(env=genv, eps=0.3, maxEpisodes=50)
print(rs)


# In[6]:


def decEpsGreedy(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  actions = []
  for i in range(maxEpisodes):
    a_s = []
    env.reset()
    reward = 0
    while env.time_elapsed < maxTime:
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))

      if np.random.rand(1) < eps:
        a = np.random.randint(env.action_space.n-1)
      else:
        a = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
      s, r, terminal, info = env.step(a)
      s, r, terminal, info = env.step(8)
      a_s.append(a)
      a_s.append(8)
      N[a] += 1
      Q[a] += (r-reward-Q[a])/N[a]
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
    actions.append(a_s)
  return Q_est, rewards, actions


# In[7]:


genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
genv.reset()
_,rs, a_s = decEpsGreedy(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', decayTill=125)
print(len(a_s))


# In[30]:


print(rs[-1])


# In[14]:


genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
genv.reset()
_,rs,_ = decEpsGreedy(env=genv, maxEpisodes=1000, eps_start=1, eps_end=0.1, decayType='exponential', decayTill=500)


# In[15]:


print(max(rs))


# In[6]:


def UCB(env, maxEpisodes, c, maxTime=300):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    count = 0
    while env.time_elapsed < maxTime:
      if count < env.action_space.n-1:
        a = count
      else:
        U = c*np.sqrt(np.log(i+1)/N)
        a = np.random.choice(np.flatnonzero(np.isclose(Q+U, np.max(Q+U))))
      s, r, terminal, info = env.step(a)
      count += 1
      s, r, terminal, info = env.step(8)
      # print(a, env.time_elapsed, r-reward)

      N[a] += 1
      Q[a] += (r-reward-Q[a])/N[a]
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[13]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = UCB(env=genv, maxEpisodes=50, c=0.3)
print(rs)


# In[7]:


def softMax(env, maxEpisodes, temp_start, temp_end, maxTime=300):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    while env.time_elapsed < maxTime:
      temp = temp_start + i*(temp_end-temp_start)/(maxEpisodes-1)
      probs = np.exp(Q/temp)/np.sum(np.exp(Q/temp))
      a = np.random.choice(a=env.action_space.n-1, p=probs)
      s, r, terminal, info = env.step(a)
      s, r, terminal, info = env.step(8)
      N[a] += 1
      Q[a] += (r-reward-Q[a])/N[a]
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[8]:


def AvgRewardsNGaussianBandits(N=50, episodes=10**3, block_type=3, decayTill=200):
    skipFirstN = 0
    pureGreedy_R = []
    pureExplore_R = []
    epsGreedy_R = []
    decExpEpsGreedy_R = []
    decLinEpsGreedy_R = []
    UCB_R = []
    softMax_R = []

    for i in tqdm(range(N)):
        genv = gym.make('foraging-replenishing-patches-v0', block_type=block_type)
        _,rs = pureGreedy(env=genv, maxEpisodes = episodes)
        pureGreedy_R.append(rs)
        _,rs = pureExplore(env=genv, maxEpisodes=episodes)
        pureExplore_R.append(rs)
        _,rs = epsGreedy(env=genv, maxEpisodes=episodes, eps=0.3)
        epsGreedy_R.append(rs)
        _,rs = decEpsGreedy(env=genv, maxEpisodes=episodes, eps_start=1, eps_end=0.1, decayType='exponential', decayTill=decayTill)
        decExpEpsGreedy_R.append(rs)
        _,rs = decEpsGreedy(env=genv, maxEpisodes=episodes, eps_start=1, eps_end=0.1, decayType='linear', decayTill=decayTill)
        decLinEpsGreedy_R.append(rs)
        _,rs = UCB(env=genv, maxEpisodes=episodes, c=0.3)
        UCB_R.append(rs)
        # _,rs = softMax(env=genv, maxEpisodes=episodes, temp_start=10**2, temp_end=0.01)
        # softMax_R.append(rs)
    
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title("Average Rewards across Time Steps for Agents in Multi Armed Bandits")
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward') 
    plt.plot(np.arange(episodes-skipFirstN), np.mean(pureGreedy_R,axis=0), label='Pure Greedy')
    plt.plot(np.arange(episodes-skipFirstN), np.mean(pureExplore_R,axis=0), label='Pure Explore')
    plt.plot(np.arange(episodes-skipFirstN), np.mean(epsGreedy_R,axis=0), label='Eps=0.3 Greedy')
    plt.plot(np.arange(episodes-skipFirstN), np.mean(decExpEpsGreedy_R,axis=0), label='Decaying (1->0.1) Exp Eps Greedy')
    plt.plot(np.arange(episodes-skipFirstN), np.mean(decLinEpsGreedy_R,axis=0), label='Decaying (1->0.1) Linear Eps Greedy')
    plt.plot(np.arange(episodes-skipFirstN), np.mean(UCB_R,axis=0), label='UCB')
    # plt.plot(np.arange(episodes), np.mean(softMax_R,axis=0), label='Soft Max')
    plt.legend()
    plt.show()  


# In[27]:


# decaying epsilon greedy, but decays till 200 episodes and constant afterwards
AvgRewardsNGaussianBandits(N=1, episodes=1000, block_type=1)


# In[9]:


#decaying epsilon greedy, but decays to final value of 0.1 till the last episode
AvgRewardsNGaussianBandits(N=1, episodes=1000, block_type=2, decayTill=500)


# In[10]:


#decaying epsilon greedy, but decays to final value of 0.1 till the last episode
AvgRewardsNGaussianBandits(N=1, episodes=1000, block_type=3, decayTill=500)


# In[33]:


#decaying epsilon greedy, but decays to final value of 0.1 till the last episode
#also averaging
AvgRewardsNGaussianBandits(N=25, episodes=200, block_type=1, decayTill=200)


# In[12]:


#decaying epsilon greedy, but decays to final value of 0.1 till the last episode
#also averaging
AvgRewardsNGaussianBandits(N=25, episodes=200, block_type=2)


# In[13]:


AvgRewardsNGaussianBandits(N=25, episodes=200, block_type=3)


# In[24]:


def RegretNGaussianBandits(N=50, episodes=10**3):
    pureGreedy_R = []
    pureExplore_R = []
    epsGreedy_R = []
    decEpsGreedy_R = []
    UCB_R = []
    softMax_R = []

    for i in tqdm(range(N)):
        genv = gym.make('foraging-replenishing-patches-v0')
        _,rs = pureGreedy(env=genv, maxEpisodes = episodes)
        pureGreedy_R.append(rs)
        _,rs = pureExplore(env=genv, maxEpisodes=episodes)
        pureExplore_R.append(rs)
        _,rs = epsGreedy(env=genv, maxEpisodes=episodes, eps=0.3)
        epsGreedy_R.append(rs)
        _,rs = decEpsGreedy(env=genv, maxEpisodes=episodes, eps_start=1, eps_end=0.0)
        decEpsGreedy_R.append(rs)
        _,rs = UCB(env=genv, maxEpisodes=episodes, c=0.5)
        UCB_R.append(rs)
        # _,rs = softMax(env=genv, maxEpisodes=episodes, temp_start=10**2, temp_end=0.01)
        # softMax_R.append(rs)
    
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title("Regret across Time Steps for Agents in Gaussian Bandits")
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Regret') 
    plt.plot(np.arange(episodes), np.mean(np.cumsum(pureGreedy_R,axis=1),axis=0), label='Pure Greedy')
    plt.plot(np.arange(episodes), np.mean(np.cumsum(pureExplore_R,axis=1),axis=0), label='Pure Explore')
    plt.plot(np.arange(episodes), np.mean(np.cumsum(epsGreedy_R,axis=1),axis=0), label='Eps Greedy')
    plt.plot(np.arange(episodes), np.mean(np.cumsum(decEpsGreedy_R,axis=1),axis=0), label='Decaying Eps Greedy')
    plt.plot(np.arange(episodes), np.mean(np.cumsum(UCB_R,axis=1),axis=0), label='UCB')
    # plt.plot(np.arange(episodes), np.mean(np.cumsum(softMax_R,axis=1),axis=0), label='Soft Max')
    plt.legend()
    plt.show() 

RegretNGaussianBandits(N=10, episodes=200)


# In[ ]:




