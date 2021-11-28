import gym
import gym_env
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def exploit_or_explore(r,T,c=0.5, beta=0.5):
  return 1 / (1 + np.exp(-(c + beta*(r - T)))) #probability of staying
    

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
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])*alpha
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards
  
  
  
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
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])*alpha
      reward = r
      a = np.argmax(Q_test)
      if a!=s : 
            leave_count[s]+=1; 
            if i==0: patch_staying_time[s].append(staying_time[s]-patch_staying_time[s][-1])
      s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards,patch_staying_time  


def sarsa_mvt(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.7, alpha = 0.2):
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
    p=0
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])*alpha
      p = exploit_or_explore(r-reward,np.max(Q_test),c=1, beta=0.5) #get probabilitity of exploiting 
      reward = r
      if np.random.rand(1) < p:continue
      else:
        a = np.argmax(Q_test)
        if a!=s : leave_count[s]+=1
        s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards
  
  
  
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
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])*alpha
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)
  return Q_est, rewards





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
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards




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
    while env.time_elapsed < maxTime:
      s, r, terminal, info = env.step(8)
      staying_time[s]+=1
      Q_test[s] += (r-reward-Q_test[s])/5
      reward = r
      a = np.argmax(Q_test)
      if a!=s : leave_count[s]+=1
      s, r, terminal, info = env.step(a)

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards




def wm_sarsa(env,maxEpisodes,eps_start, eps_end, decayType='exponential', maxTime=300, decayTill=200, gamma = 0.5, alpha = 0.5, forgetDecay=0.1, capacity = 5, load =False):
  Q_est = np.zeros((maxEpisodes,env.action_space.n-1))
  rewards = []
  Q1 = np.zeros(env.action_space.n-1)
  N = np.zeros(env.action_space.n-1)
  Q2 = np.zeros(env.action_space.n-1)
  if env.block_type==1: L=5
  if env.block_type==2: L=6
  if env.block_type==3: L=8
  if not load : L=8
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

  avg_harvest_time=(np.sum(staying_time))/10
  avg_travel_time=300 - avg_harvest_time
  avg_staying_time = staying_time/leave_count
  rewards=(rewards,avg_harvest_time,avg_staying_time)

  return Q_est, rewards

