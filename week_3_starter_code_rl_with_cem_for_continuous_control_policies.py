# -*- coding: utf-8 -*-
"""Week 3: Starter Code - RL with CEM for Continuous Control Policies

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t6RlpLSU0dg7fJW5PrUJYKOu2WNipwid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation, rc, image
import seaborn as sns
import math
from math import sin, cos

#Fixed parameters
in_size = 5     #Number of observations (pos_x, pos_y, theta, goal_x, goal_y)
out_size = 2    #Number of o

#Policy parameters
hidden_size = 5        # How many values in the hidden layer
evalation_samples = 1 # How many samples to take when evaluating a network

#Training parameters
cem_iterations = 100    # How many total CEM iterations 
cem_batch_size = 50     # How many guassian samples in each CEM iteration
cem_elite_frac = 0.5    # What percentage of cem samples are used to fit the guassian for next iteration
cem_init_stddev = 1.0   # Initial CEM guassian uncertainty
cem_noise_factor = 1.0    # Scaling factor of how much extra noise to add each iteration (noise_factor/iteration_number noise is added to std.dev.)
cem_print_rate = 5

#Simulation paramters
dt = 0.1    #seconds
runtime = 8 #seconds

#Target task
car_start = np.array((-50,0,0.751))
car_goal = np.array((50,0))

#Car dynamics paramters
v_max = 80  #units/sec
omega_max = 3.14 #pi radians/sec = 180 deg/sec turn speed

#Car shape
car_w = 5
car_l = 10

linear_policy_size = (in_size+1)*out_size
def linear_model(params, in_data):
  in_vec = np.array(in_data).reshape(in_size,1) #place input data in a column vector 

  m1_end = in_size*out_size
  matrix1 = np.reshape(params[0:m1_end], (out_size,in_size))
  biases1 = np.reshape(params[m1_end:m1_end + out_size], (out_size,))
  result = (matrix1 @ in_vec) + biases1
  return result

cur_Model = linear_model
policy_size = linear_policy_size

two_layer_policy_size = (in_size+1)*hidden_size + (hidden_size+1)*out_size
def two_layer_model(params, in_data):
  in_vec = np.array(in_data).reshape(in_size,1) #place input data in a column vector

  #Layer 1 (input -> hidden)
  m1_end = hidden_size*in_size
  matrix1 = np.reshape(params[0:m1_end], (hidden_size,in_size))
  biases1 = np.reshape(params[m1_end:m1_end+hidden_size], (hidden_size,1))
  hidden_out = (matrix1 @ in_vec) + biases1
  hidden_out = hidden_out * (hidden_out > 0) + 0.1*hidden_out * (hidden_out < 0) #Leaky ReLU

  #Layer 2 (hiden -> output)
  m2_start = m1_end+hidden_size
  m2_end = m2_start + out_size*hidden_size
  matrix2 = np.reshape(params[m2_start:m2_end], (out_size,hidden_size))
  biases2 = np.reshape(params[m2_end:m2_end+out_size], (out_size,1))
  result = (matrix2 @ hidden_out) + biases2
  return result.reshape(out_size,)

cur_Model = two_layer_model
policy_size = two_layer_policy_size

#The the input paramters theta that maximize the function f
#CEM will extimate a guassian of the optimal theta represented as
#  th_mean (the means) and th_std (the standard deviation).
#This CEM implementation, has one value of std.dev. for each paramter, and
#  ignores off-diagnol terms of the co-variance matrix
def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
  n_elite = int(np.round(batch_size*elite_frac))
  th_std = np.ones_like(th_mean) * initial_std

  for iter in range(n_iter):
    #Add noise to batch_size samples 
    ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
    #Evaluate each sample
    ys = np.array([f(th,evalation_samples) for th in ths])
    #Keep top n_elite best samples
    elite_inds = ys.argsort()[::-1][:n_elite]
    elite_ths = ths[elite_inds]
    #Compute the mean and std-dev of best samples
    th_mean = elite_ths.mean(axis=0)
    th_std = elite_ths.std(axis=0)
    #Add some extra noise
    th_std += cem_noise_factor/(iter+1)
    #Return results 
    yield {'ys' : ys, 'theta_mean' : th_mean, 'theta_std' : th_std, 'y_mean' : ys.mean(), 'f_th_mean' : f(th_mean,100), 'mean_of_std_dev': th_std.mean()}

def update_state(params, cur_state, goal):
  #Read Current State
  cx = cur_state[0]
  cy = cur_state[1]
  theta = cur_state[2]
  gx = goal[0]
  gy = goal[1]

  #Apply Policy
  action = cur_Model(params, (cx, cy, theta, gx, gy))
  v = action[0]
  omega = action[1]

  #Clamp actions
  v = np.clip(v,-v_max,v_max)
  omega = np.clip(omega, -omega_max, omega_max)

  #Apply dynamic model (Eulerian Integration)
  vx = v * np.cos(theta)
  vy = v * np.sin(theta)

  #Integrate
  theta += omega*dt
  cx += vx * dt
  cy += vy *dt
  return (cx, cy, theta), action   #new state, action

def run_model(params, init_state, goal_pos):
  state_list = []
  action_list = []

  cur_state = init_state
  state_list.append(cur_state)

  sim_time = runtime
  reward = 0
  for i in range(int(sim_time/dt)):
    cur_state, new_action = update_state(params, cur_state, goal_pos)
    state_list.append(cur_state)
    action_list.append(new_action)
    
  #print(action_list)
  return state_list,action_list

def reward(policy, num_tasks = 10):
  total_reward = 0
  for _ in range(num_tasks):
    #Run task
    init_state = car_start
    goal_pos = car_goal
    states, actions = run_model(policy, init_state, goal_pos)

    #Compute reward
    task_reward = 0
    for i in range(len(actions)):
      cur_state = states[i+1]
      cur_action = actions[i]
      dx = cur_state[0] - goal_pos[0]
      dy = cur_state[1] - goal_pos[1]
      dist = math.sqrt(dx*dx + dy*dy)
      #accumulate intermediate reward
      task_reward -= dist
      task_reward -= 1*abs(cur_action[0])
      task_reward -= 1*abs(cur_action[1]) #penalize large rotational velocities
    #Final state bonus
    final_state = states[-1]
    final_action = actions[-1]
    final_dist = dist
    if final_dist < 20: task_reward += 1000
    if final_dist < 10 and abs(final_action[0]) < 5: task_reward += 10000
    total_reward += task_reward

  return total_reward/num_tasks

init_params = np.zeros(policy_size)
cem_params = dict(n_iter=cem_iterations, batch_size=cem_batch_size, elite_frac=cem_elite_frac, initial_std=cem_init_stddev)
for (i, iterdata) in enumerate(
    cem(reward, init_params, **cem_params)):
    if (i+1)%cem_print_rate == 0 or i == 0: print('Iter: %3i  mean(reward): %6.3f  f(theta_mean): %6.3f  avg(std_dev):%6.3f'%(i+1, iterdata['y_mean'],iterdata['f_th_mean'],iterdata['mean_of_std_dev']))

mean_policy = iterdata['theta_mean']
policy_std = iterdata['theta_std']

gx = 50
gy = 0
states,actions = run_model(mean_policy, (-50,0,0.751),(gx,gy))
for s in actions:
  print(s)

render_w = 200
render_h = 200

frames_to_draw = len(states)
print("Rendering",frames_to_draw,"frames")

#Matplot lib animation magic...
#Check here for a lot more: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

fig = plt.figure()
ax = plt.gca()

def init():
  ax.cla()
  ax.axis('equal')
  ax.set(xlim=(-render_w, render_w),ylim=(-render_h, render_h))
  return fig,

def animate(i):
  ax.cla()
  ax.axis('equal')
  #Draw area
  ax.set(xlim=(-render_w, render_w),ylim=(-render_h, render_h))
  #Draw goal
  goal_w = car_l
  rect = Rectangle((gx-goal_w/2,gy-goal_w/2),goal_w,goal_w,linewidth=2,edgecolor='g',facecolor='g')
  #Draw agent
  ax.add_patch(rect)
  cx = states[i][0]
  cy = states[i][1]
  theta = states[i][2]
  ang = np.rad2deg(theta)
  rect = Rectangle((cx+car_w/2*sin(theta),cy-car_w/2*cos(theta)),car_l,car_w,angle=np.rad2deg(theta),linewidth=2,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  return fig,

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=frames_to_draw,interval=dt*1000,repeat_delay=1500,blit=True)
rc('animation', html='jshtml')
anim