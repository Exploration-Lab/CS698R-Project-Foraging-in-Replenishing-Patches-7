#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install --upgrade --editable ../gym-env')


# In[1]:


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


# In[2]:


def pureGreedy(env,maxEpisodes, maxTime=300, optimistic=True, forgetDecay=0.5):
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
        Q *= (1-forgetDecay)
        Q[a] /= (1-forgetDecay)
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[11]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureGreedy(env=genv, maxEpisodes=50, optimistic=True)
print(rs)


# In[12]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureGreedy(env=genv, maxEpisodes=50, optimistic=False)
print(rs)


# In[15]:


for i in tqdm(range(0,10,2)):
  RS = []
  for _ in range(5):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = pureGreedy(env=genv, maxEpisodes=200, optimistic=False, forgetDecay=i/10)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[3]:


def pureExplore(env,maxEpisodes, maxTime=300, forgetDecay=0.5):
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
        Q *= (1-forgetDecay)
        Q[a] /= (1-forgetDecay)
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[17]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = pureExplore(env=genv, maxEpisodes=50)
print(rs)


# In[18]:


for i in tqdm(range(0,10,2)):
  RS = []
  for _ in range(5):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = pureExplore(env=genv, maxEpisodes=200, maxTime=300, forgetDecay=i/10)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[4]:


def epsGreedy(env,eps, maxEpisodes,maxTime=300, forgetDecay=0.5):
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
        Q *= (1-forgetDecay)
        Q[a] /= (1-forgetDecay)        
        reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[20]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = epsGreedy(env=genv, eps=0.3, maxEpisodes=50)
print(rs)


# In[3]:


def decEpsGreedy(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, forgetDecay=0.5):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
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
      N[a] += 1
      Q[a] += (r-reward-Q[a])/N[a]
      Q *= (1-forgetDecay+1e-9)
      Q[s] /= (1-forgetDecay+1e-9)
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[22]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = decEpsGreedy(env=genv, maxEpisodes=50, eps_start=1, eps_end=0.1, decayType='linear', decayTill=25)
print(rs)


# In[23]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = decEpsGreedy(env=genv, maxEpisodes=50, eps_start=1, eps_end=0.1, decayType='exponential', decayTill=10)
print(rs)


# In[6]:


for i in tqdm(range(0,10,2)):
  RS = []
  for _ in range(5):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = decEpsGreedy(env=genv, maxEpisodes=250, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125, forgetDecay=(i/10))
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward')
  if i==0: 
    plt.plot(np.arange(250), np.mean(RS,axis=0), label=str(i/10))
  else:
    plt.plot(np.arange(250), np.mean(RS,axis=0), label=str(np.round(1-i/10,1)))
  plt.legend()

plt.show()


# In[25]:


def UCB(env, maxEpisodes, c, maxTime=300, forgetDecay=0.5):
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
      N[a] += 1
      Q[a] += (r-reward-Q[a])/N[a]
      Q *= (1-forgetDecay)
      Q[a] /= (1-forgetDecay)      
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[26]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = UCB(env=genv, maxEpisodes=50, c=0.3, forgetDecay=0.7)
print(rs)


# In[27]:


for i in tqdm(range(0,10,2)):
  RS = []
  for _ in range(5):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = UCB(env=genv, maxEpisodes=200, c=0.3, maxTime=300, forgetDecay=i/10)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[29]:


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


# In[32]:


#decaying epsilon greedy, but decays to final value of 0.1 till the last episode
AvgRewardsNGaussianBandits(N=1, episodes=1000, block_type=1, decayTill=1000)


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




