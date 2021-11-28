#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade --editable gym-env')


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


# In[10]:


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
      Q *= (1-forgetDecay)
      Q[a] /= (1-forgetDecay)
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[12]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = decEpsGreedy(env=genv, maxEpisodes=50, eps_start=1, eps_end=0.1, decayType='exponential', decayTill=10)
print(rs)


# In[22]:


for i in tqdm(range(0,11,5)):
  RS = []
  for _ in range(5):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = decEpsGreedy(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125, forgetDecay=i/10)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# Suggestions
# 
# 1. decay factor as a function of time
# 2. decay Q for patches not on currently which were last visited for some tau value
# 3. recency effect (initial patches have had more time to be recorded, for last patches more recency)
#     a. knows current patch
#     b. forgets patches not visited for a while
# 
# 4. Working Memory
# 
# 5. Long Term Working Memory
#     a. identify non rewarding patches this has lower decay factor because lesser information needed to store 
#     b. but decay memory of rewarding patches with a greater decay factor because more information needed to store
# 
# 6. Distance Heuristics
#     a. reward = reward + reward_distance
#     b. opportunity cost of patch foraging paper, eq3
# 
# 7. MDP Approaches
# 
# 8. Marginal Value Theorem
#     a. Can't use directly because revisiting allowed in our game
#     b. One way for global average: (total reward in env) / (total time so far)
#     c. Local Reward: (total patch reward) / (total time so far in that patch)
#         i. this matters in patch leaving
#         ii. determines policy for patch leaving (test for this)
#     d. Varies temporally
#     e. Include travel time in MVT
# 
#     

# In[20]:


def sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.9, alpha = 0.5):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
      _, r, terminal, info = env.step(s)
      _, r, terminal, info = env.step(8)
      target = r - reward
      if not terminal : target = target + gamma*Q[ns]
      error = target - Q[s]
      Q[s] = Q[s]+alpha*error
      s=ns
      #Q *= (1-forgetDecay)
      #Q[a] /= (1-forgetDecay)
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[21]:


genv = gym.make('foraging-replenishing-patches-v0')
genv.reset()
_,rs = sarsa(env=genv, maxEpisodes=50, eps_start=1, eps_end=0.1, decayType='linear', decayTill=25)
print(rs)


# In[38]:



RS = []
for _ in range(10):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = sarsa(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125)
    RS.append(rs)
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(200), np.mean(RS,axis=0))
plt.legend()

plt.show()


# In[27]:


def sarsa_lda(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.9, alpha = 0.5, lda=0.5):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  E = np.zeros(env.action_space.n-1)
    
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
      _, r, terminal, info = env.step(s)
      _, r, terminal, info = env.step(8)
      target = r - reward
      if not terminal : target = target + gamma*Q[ns]
      error = target - Q[s]
      E[s] +=1 
      Q = Q + alpha*error*E
      E=gamma*lda*E
      s=ns
      #Q *= (1-forgetDecay)
      #Q[a] /= (1-forgetDecay)
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[30]:


for i in tqdm(range(0,11,5)):
  RS = []
  for _ in range(10):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = sarsa_lda(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125, lda=i/10)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[35]:


def forgetfull_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.9, alpha = 0.5, forgetDecay=0.1):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  E = np.zeros(env.action_space.n-1)
    
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
      _, r, terminal, info = env.step(s)
      _, r, terminal, info = env.step(8)
      target = r - reward
      if not terminal : target = target + gamma*Q[ns]
      error = target - Q[s]
      Q = Q + alpha*error
      Q *= (1-forgetDecay)
      Q[s] /= (1-forgetDecay)
      s=ns
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
  return Q_est, rewards


# In[37]:


for i in tqdm(range(0,11,5)):
  RS = []
  for _ in range(10):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = forgetfull_sarsa(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125, forgetDecay=i/11)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[44]:


def forgetDecay_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.9, alpha = 0.5, forgetfullness = 0.1, forgetDecay=0.1):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  E = np.zeros(env.action_space.n-1)
  f = forgetfullness
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.random.choice(np.flatnonzero(np.isclose(Q, np.max(Q))))
      _, r, terminal, info = env.step(s)
      _, r, terminal, info = env.step(8)
      target = r - reward
      if not terminal : target = target + gamma*Q[ns]
      error = target - Q[s]
      Q = Q + alpha*error
      Q *= (1-f)
      Q[s] /= (1-f)
      s=ns
      reward = r
    rewards.append(reward)
    Q_est[i] = Q
    f=f*forgetDecay
  return Q_est, rewards


# In[45]:


for i in tqdm(range(0,11,5)):
  RS = []
  for _ in range(10):
    genv = gym.make('foraging-replenishing-patches-v0')
    genv.reset()
    _,rs = forgetDecay_sarsa(env=genv, maxEpisodes=200, eps_start=1, eps_end=0.1, decayType='exponential', maxTime=300, decayTill=125, forgetfullness=i/11)
    RS.append(rs)
  plt.rcParams["figure.figsize"] = (20,10)
  plt.title("Rewards across Time Steps for Agents in Multi Armed Bandits on varying forget decay")
  plt.xlabel('Episodes')
  plt.ylabel('Reward') 
  plt.plot(np.arange(200), np.mean(RS,axis=0), label=str(i/10))
  plt.legend()

plt.show()


# In[ ]:




