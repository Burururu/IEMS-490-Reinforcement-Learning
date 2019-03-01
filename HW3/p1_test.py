# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:10:16 2018

@author: Bruce
"""


import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt


import random
import time

import os
if 'deepdish' in os.path.abspath('.'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])


class DQN():
    def __init__(self, scope_name='tmp'):
        self.num_actions = 4
        with tf.variable_scope(scope_name):
            self.build_network()
        
    def build_network(self):
        # placeholders
        self.X = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.uint8, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        batch_size = tf.shape(self.X)[0]
        
        # network structure
        self.conv1 = tf.contrib.layers.conv2d(inputs=tf.to_float(self.X), num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        self.conv2 = tf.contrib.layers.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        self.conv3 = tf.contrib.layers.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=2, stride=1, activation_fn=tf.nn.relu)        
        conv3_shape = int(self.conv3.shape[1]*self.conv3.shape[2]*self.conv3.shape[3])
        self.fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.conv3, [-1, conv3_shape]), 1024)
        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 128)

        # qvalue for all actions
        self.output = tf.contrib.layers.fully_connected(self.fc2, 4)
        
        # q_value for given action
        indices = tf.range(batch_size) * tf.shape(self.output)[1] + self.action
        self.q_value = tf.gather(tf.reshape(self.output, [-1]), indices)
        
        # calculate loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.q_value))
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
    def predict(self, sess, X):
        return sess.run(self.output, feed_dict={self.X: X})
    
    def update(self, sess, current_state, actions, target_y):
        feed_dict= {self.X: current_state, self.y: target_y, self.action: actions}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict)
        return loss

    
def generate_epsilon_greedy_policy(sess, eps, q_network, X):
    num_actions = q_network.num_actions
    q_values = q_network.predict(sess, X)
    res = np.ones(num_actions) * eps / num_actions
    res[np.argmax(q_values)] = 1 - (num_actions-1)/num_actions*eps
    return np.random.choice(q_network.num_actions, p=res)

def update_target_network(sess):
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q')
    
    with tf.variable_scope('soft_replacement'):
        target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
    sess.run(target_replace_op)
    
def record(lst):
    return ';'.join(map(str, lst))+'\n'
    

env = gym.make('Breakout-v0')
env = env.unwrapped

num_episode = 10
num_actions = env.action_space.n

eps_max = 1
eps_min = 0.1
eps_decay_rate = 1e-5

q_network = DQN('q')
target_network = DQN('target')


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './checkpoint/model')

# main
global_step_counter = 0
episode_rewards_list = []
for episode_counter in range(num_episode):
    # setup some statistics for the experiments      
    episode_step_counter = 0
    episode_reward = 0
    episode_max_q_value = -1
    current_remain_lives = 5
    next_remain_lives = 5
    
    # start a new game
    img = preprocess(env.reset())
    current_state = np.stack([img, img, img, img], axis=2)
    
    while True:
        # sample the action by eps greedy policy
        eps = 0.1
        action = generate_epsilon_greedy_policy(sess, eps, q_network, current_state.reshape(1, 80, 80, 4))
        
        # if you lost a live, you must fire a new ball immedietely
        if current_remain_lives != next_remain_lives:
            action = 1
            current_remain_lives = next_remain_lives
        
        # make a step for the game
        img, reward, done, next_remain_lives = env.step(action)
        next_remain_lives = next_remain_lives['ale.lives']
        
        # calculate the state for next step and update replay buffer
        img = preprocess(img).reshape(80, 80, 1)
        next_state = np.concatenate([current_state[:,:,1:], img], axis=2)            

        # update state and statistics
        global_step_counter += 1
        episode_step_counter += 1
        episode_reward += reward
        prob = q_network.predict(sess, current_state.reshape(1,80,80,4))[0].tolist()
        episode_max_q_value = max(episode_max_q_value, np.max(prob))
        current_state = next_state

        # stop the while loop if game is done
        if done:    
            break
    
    episode_rewards_list.append(episode_reward)
    print(episode_counter, episode_reward)

fig = plt.figure()
plt.plot(episode_rewards_list)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('p1.png')
    


q_value_list = []
file = open('log_episodes.txt', 'r')
for line in file.readlines():
    res = line[:-1].split(';')[-1]
    q_value_list.append(float(res))

file.close()

fig = plt.figure()
plt.plot(q_value_list)
plt.xlabel('episode')
plt.ylabel('max q-value')
plt.savefig('p1_qvalue.png')
    
