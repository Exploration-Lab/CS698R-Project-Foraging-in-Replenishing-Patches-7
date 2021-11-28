#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
    #print(action, state, reward, done)
    if done:
        break


# In[26]:


def exploit_or_explore(r,T,c=0.5, beta=0.5):

    return 1 / (1 + np.exp(-(c + beta*(r - T)))) #probability of staying


# In[27]:


def sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.7, alpha = 0.2):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*(Q[ns])
      error = target - Q[s]
      alpha = 1 + min(decayTill/1.6 -1,i)*(0.2-1)/(decayTill/1.6-1)
      Q[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
      
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[28]:


def q_learning(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.7, alpha = 0.2):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*np.max(Q)
      error = target - Q[s]
      alpha = 1 + min(decayTill/1.6 -1,i)*(0.2-1)/(decayTill/1.6 -1)
      Q[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
      
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  patch_staying_time=[[0],[0],[0],[0],[0],[0],[0],[0]]
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1; patch_staying_time[s].append(staying_time[s]-patch_staying_time[s][-1])
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards,patch_staying_time


# In[43]:


def q1_learning(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.7, alpha = 0.2):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*np.max(Q)
      error = target - Q[s]
      alpha = 1 + min(decayTill/1.6 -1,i)*(0.2-1)/(decayTill/1.6 -1)
      Q[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
      
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  patch_staying_time=[[0],[0],[0],[0],[0],[0],[0],[0]]
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : 
            leave_count[s]+=1; 
            if i==0: patch_staying_time[s].append(staying_time[s]-patch_staying_time[s][-1])
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards,patch_staying_time


# In[29]:


def sarsa_mvt(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.7, alpha = 0.7):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  p=0
  N=np.zeros(env.action_space.n-1)
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.uniform() < p:
        ns=env.current_state  
      elif np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*Q[ns]
      error = target - Q[s]
      Q[s] += error/N[s]
      p = exploit_or_explore(r-reward,np.max(Q),c=2, beta=0.3) #get probabilitity of exploiting
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    p=0
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      p = exploit_or_explore(r-reward,np.max(Q_test),c=1, beta=0.5) #get probabilitity of exploiting 
      reward = r
      if np.random.rand(1) < p:continue
      else:
        a = np.argmax(Q_test)
        if a!=s : leave_count[s]+=1
        s, r, terminal, info = env.step(a)
        
     #plot_rewards.append(reward)
    #plt.plot(plot_rewards)
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[30]:


def sarsa_lda(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.4, lda=0.5):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  E = np.zeros(env.action_space.n-1)

  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*(Q[ns])
      error = target - Q[s]
      E[s] +=1
      Q += error*E*alpha
      E=gamma*lda*E
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
      
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  #plt.legend()
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)
  return Q_est, rewards


# In[31]:


avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
print('\n\nSARSA')
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.title("sarsa epsilon decay [1->0 till 500 eps]")
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa')


print('\n\nQlearning')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = q_learning(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='q_learing')

print('\n\nSARSA MVT')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = sarsa_mvt(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa_mvt')

print('\n\nSARSA LDA')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = sarsa_lda(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.1, decayType='linear', maxTime=300, decayTill=350, lda=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa_lda')
plt.title("MDP Agents Gamma=0.5 Lamba=0.5 Eps=[1->0]")

plt.legend()
plt.savefig('basic_mdp_stable.pdf')


# In[32]:


def forgetfull_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.2, forgetDecay=0.1):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N=np.zeros(env.action_space.n-1)
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*(Q[ns])
      error = target - Q[s]
      Q[s] += error/N[s]
      Q *=(1-forgetDecay)
      Q[s]/=(1-forgetDecay)
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q
    
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[33]:


def forgetDecay_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.5, forgetDecay=0.1):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q = np.zeros(env.action_space.n-1)
  N=np.zeros(env.action_space.n-1)
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
    
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        ns = np.argmax(Q)
      target = r - reward
      if not terminal : target = target + gamma*(Q[ns])
      error = target - Q[s]
      Q[s] += error/N[s]
      Q *=(1-forgetDecay)
      Q[s]/=(1-forgetDecay)
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q
    forgetDecay=forgetDecay*0.5
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q_test = np.array(Q)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[54]:


def wm_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.5, forgetDecay=0.1, capacity = 0.8):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q1 = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  Q2 = np.zeros(env.action_space.n-1)

  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
      forget = False
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        if np.random.rand(1) < capacity:
            ns = np.argmax(Q1)
        else:
            forget = True
            ns = np.argmax(Q2)
      target = r - reward
      if not terminal : target = target + gamma*Q1[ns]
      error = target - Q1[s]
      if forget:
            Q2[s] += error/N[s]
            Q2 *= (1-forgetDecay)
            Q2[s] /= (1-forgetDecay)
      else:
            Q1[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q1
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q1_test = np.array(Q1)
    Q2_test = np.array(Q2)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q1_test[s] += (r-reward-Q1_test[s])/5
      reward = r
      if np.random.rand(1) < capacity:
            a = np.argmax(Q1_test)
      else:
            forget = True
            a = np.argmax(Q2_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[35]:


print('\n\nSARSA 0 Forgetful')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = forgetfull_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350,forgetDecay=0)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='no forget sarsa')

print('\n\nSARSA 0.5 Forgetful')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = forgetfull_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350,forgetDecay=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='forgetful sarsa')

print('\n\n Sarsa Forget Decay')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = forgetDecay_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='forgetdecay sarsa')

print('\n\nSARSA WM 5/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 5/8)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa capacity = 5/8')

print('\n\nSARSA WM 7/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 7/8)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa capacity = 7/8')
plt.title("MDP Agents With Forgetfulness")

plt.legend()
plt.savefig('forget_mdp.pdf')


# In[36]:


avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
print('\n\nSARSA')
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.title("sarsa epsilon decay [1->0 till 500 eps]")
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa')


print('\n\nQlearning')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = q_learning(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='q_learing')

print('\n\nSARSA MVT')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = sarsa_mvt(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa_mvt')

print('\n\nSARSA LDA')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = sarsa_lda(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.1, decayType='linear', maxTime=300, decayTill=350, lda=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='sarsa_lda')
plt.title("MDP Agents Gamma=0.5 Lamba=0.5 Eps=[1->0]")

plt.legend()
plt.savefig('basic_mdp_stable_b1.pdf')


# In[37]:


print('\n\nSARSA 0 Forgetful')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = forgetfull_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350,forgetDecay=0)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='no forget sarsa')

print('\n\nSARSA 0.5 Forgetful')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = forgetfull_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350,forgetDecay=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='forgetful sarsa')

print('\n\n Sarsa Forget Decay')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = forgetDecay_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='forgetdecay sarsa')

print('\n\nSARSA WM 5/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 5/8)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa capacity = 5/8')

print('\n\nSARSA WM 7/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 7/8)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa capacity = 7/8')
plt.title("MDP Agents With Forgetfulness")

plt.legend()
plt.savefig('forget_mdp_b1.pdf')


# In[44]:


avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs, patch = q1_learning(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('timestep')
plt.ylabel('stay time') 
for i in range (8):
    plt.plot(patch[i])


# In[45]:


for i in range (8):
    plt.plot(patch[i], label=str(i))
plt.legend()


# In[47]:


avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs, patch = q1_learning(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, gamma=0.5, alpha=0.2)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('timestep')
plt.ylabel('stay time') 
for i in range (8):
    plt.plot(patch[i], label=str(i))
plt.legend()


# In[57]:


def wm1_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.5, forgetDecay=0.1, capacity = 0.8):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q1 = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  Q2 = np.zeros(env.action_space.n-1)

  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
      forget = False
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        if np.random.rand(1) < capacity:
            ns = np.argmax(Q1)
        else:
            forget = True
            ns = np.argmax(Q2)
      target = r - reward
      if not terminal : target = target + gamma*Q1[ns]
      error = target - Q1[s]
      if forget:
            Q2[s] += error/N[s]
            Q2 *= (1-forgetDecay)
            Q2[s] /= (1-forgetDecay)
      else:
            Q1[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q1
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  patch_staying_time=[[0],[0],[0],[0],[0],[0],[0],[0]]
  for i in range(10):
    Q1_test = np.array(Q1)
    Q2_test = np.array(Q2)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q1_test[s] += (r-reward-Q1_test[s])/5
      reward = r
      if np.random.rand(1) < capacity:
            a = np.argmax(Q1_test)
      else:
            forget = True
            a = np.argmax(Q2_test)
      if a!=s : 
        leave_count[s]+=1
        if i==0: 
            patch_staying_time[s].append(staying_time[s]-patch_staying_time[s][-1])
            for k in range(8):
                if k==s: continue;
                patch_staying_time[k].append(0)
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards, patch_staying_time


# In[58]:


avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs, patch = wm1_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 5/8)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('nth visit')
plt.ylabel('stay time') 
for i in range (8):
    plt.plot(patch[i], label=str(i))
plt.legend()


# In[52]:


def wm2_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.5, forgetDecay=0.1, capacity = 5):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q1 = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  Q2 = np.zeros(env.action_space.n-1)
  if env.block_type==1: L=5
  if env.block_type==2: L=6
  if env.block_type==3: L=8
  for i in range(maxEpisodes):
    env.reset()
    reward = 0
    s=env.current_state
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      N[s]+=1
      forget = False
      if decayType == 'linear':
        eps = eps_start + min(decayTill-1,i)*(eps_end-eps_start)/(decayTill-1)
      else:
        eps = eps_start*((eps_end/eps_start)**(min(decayTill-1,i)/(decayTill-1)))
      if np.random.rand(1) < eps:
        ns = np.random.randint(env.action_space.n-1)
      else:
        if np.random.rand(1) < capacity/L:
            ns = np.argmax(Q1)
        else:
            forget = True
            ns = np.argmax(Q2)
      target = r - reward
      if not terminal : target = target + gamma*Q1[ns]
      error = target - Q1[s]
      if forget:
            Q2[s] += error/N[s]
            Q2 *= (1-forgetDecay)
            Q2[s] /= (1-forgetDecay)
      else:
            Q1[s] += error/N[s]
      s=ns
      reward = r
      _, r, terminal, info = env.step(s)
    rewards.append(reward)
    Q_est[i] = Q1
  staying_time=np.zeros(env.action_space.n-1)
  leave_count=np.zeros(env.action_space.n-1)
  
  for i in range(10):
    Q1_test = np.array(Q1)
    Q2_test = np.array(Q2)
    env.reset()
    reward = 0
    plot_rewards = []
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q1_test[s] += (r-reward-Q1_test[s])/5
      reward = r
      if np.random.rand(1) < capacity:
            a = np.argmax(Q1_test)
      else:
            forget = True
            a = np.argmax(Q2_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)
      plot_rewards.append(reward)
    #plt.plot(plot_rewards,label =str(i))
  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards


# In[56]:


print('\n\nSARSA WM 7/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=1')
plt.title("MDP Agents With Forgetfulness")

avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=2)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=2')
plt.title("MDP Agents With Forgetfulness")

avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = wm_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=3')
plt.title("MDP Agents With Forgetfulness")
print('\n\nSARSA WM 7/8')
avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=1)
    genv.reset()
    _,rs = wm2_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=1 with load=5')
plt.title("MDP Agents With Forgetfulness")

avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=2)
    genv.reset()
    _,rs = wm2_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=2 with load=6')
plt.title("MDP Agents With Forgetfulness")

avg_harvest_time=0
avg_staying_time=np.zeros(8)
max_reward = 0
RS = []
for _ in range(20):
    genv = gym.make('foraging-replenishing-patches-v0', block_type=3)
    genv.reset()
    _,rs = wm2_sarsa(env=genv, maxEpisodes=500, eps_start=1, eps_end=0.01, decayType='linear', maxTime=300, decayTill=350, forgetDecay=0.5, capacity = 4)
    RS.append(rs[0])
    max_reward+=rs[0][-1]
    avg_harvest_time+=rs[1]
    avg_staying_time+=rs[2]
print('max_reward',max_reward/20)
print('avg_harvest_time', avg_harvest_time/20)
print('avg_travel_time',300 - avg_harvest_time/20)
print('avg_staying_time',avg_staying_time/20)
plt.rcParams["figure.figsize"] = (20,10)
plt.xlabel('Episodes')
plt.ylabel('Reward') 
plt.plot(np.arange(500), np.mean(RS,axis=0), label='wm_sarsa block=3 with load=8')
plt.title("MDP Agents With Forgetfulness")

plt.legend()
plt.savefig('wm_b.pdf')


# In[ ]:




