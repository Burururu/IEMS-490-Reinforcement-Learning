#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:49:40 2018

@author: Bruce
"""




import gym
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

import matplotlib.pyplot as plt
#%matplotlib inline

mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80)


class DQN():
    def __init__(self, scope_name='tmp'):
        self.num_actions = 9
        with tf.variable_scope(scope_name):
            self.build_network()
        
    def build_network(self):
        # placeholders
        self.X = tf.placeholder(shape=[None, 88, 80, 4], dtype=tf.int8, name="X")
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
        self.output = tf.contrib.layers.fully_connected(self.fc2, 9)
        
        # q_value for given action
        indices = tf.range(batch_size) * tf.shape(self.output)[1] + self.action
        self.q_value = tf.gather(tf.reshape(self.output, [-1]), indices)
        
        # calculate loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.q_value))
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
             
    def predict(self, sess, X):
        return sess.run(self.output, feed_dict={self.X: X})
    
    def update(self, sess, s, a, y):
        feed_dict= {self.X: s, self.y: y, self.action: a}
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


replay_buffer_size = 1e5
num_episode = 10000
discount_factor = 0.99
batch_size = 128
num_actions = 9
num_episode = 10000

eps_max = 1
eps_min = 0.1
eps_decay_rate = 1e-5



q_network = DQN('q')
target_network = DQN('target')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

env = gym.envs.make("MsPacman-v0")
env = env.unwrapped


#try:
#    saver = tf.train.Saver()
#    saver.restore(sess, './checkpoint/model')
#except:
#    print('None existinSg checkpoint')


# initialize replay buffer
img = preprocess_observation(env.reset())
current_state = np.stack([img, img, img, img], axis=2)
print('Init replay buffer...')
replay_buffer = []
for replay_counter in range(10000):
    action = generate_epsilon_greedy_policy(sess, 0.4, q_network, current_state.reshape(1, 88, 80, 4))
    img, reward, done, _ = env.step(action)
    img = preprocess_observation(img).reshape(88, 80, 1)
    next_state = np.concatenate([current_state[:,:,1:], img], axis=2)
    replay_buffer.append([action, reward, done, current_state, next_state])
    if done:
        img = preprocess_observation(env.reset())
        current_state = np.stack([img, img, img, img], axis=2)
    else:
        current_state = next_state
print('Init replay buffer finished...')

# main
global_step_counter = 0
for episode_counter in range(num_episode):
    # setup some statistics for the experiments
    episode_step_counter = 0
    episode_reward = 0
    episode_max_q_value = -1
    
    # start a new game
    img = preprocess_observation(env.reset())
    current_state = np.stack([img, img, img, img], axis=2)
    
    while True:
        # sample the action by eps greedy policy
        action = generate_epsilon_greedy_policy(sess, 0.1, q_network, current_state.reshape(1, 88, 80, 4))
        
        # make a step for the game
        img, reward, done, next_remain_lives = env.step(action)
        
        # calculate the state for next step and update replay buffer
        img = preprocess_observation(img).reshape(88, 80, 1)
        next_state = np.concatenate([current_state[:,:,1:], img], axis=2)
        replay_buffer.append([action, reward, done, current_state, next_state])
        
        # keep the replay buffer in given size
        if len(replay_buffer) > replay_buffer_size:
            replay_buffer.pop(0)

        # get batches from replay buffer
        batch_history = random.sample(replay_buffer, batch_size)
        action_batch, reward_batch, done_batch, current_states_batch, next_states_batch = [np.array([batch_history[i][j] for i in range(batch_size)]) for j in range(len(batch_history[0]))]
        
        # update q_network
        target_q_values_next = target_network.predict(sess, next_states_batch)
        target_y_batch = reward_batch + discount_factor*np.max(target_q_values_next, axis=1)*(1-done_batch)
        loss = q_network.update(sess, current_states_batch, action_batch, target_y_batch)
    
        # update state and statistics
        global_step_counter += 1
        episode_step_counter += 1
        episode_reward += reward
        q_value_all_actions = q_network.predict(sess, current_state.reshape(1,88,80,4))[0].tolist()
        episode_max_q_value = max(episode_max_q_value, np.max(q_value_all_actions))
        current_state = next_state
        
        # update target_network
        if global_step_counter % 100 == 0:
            update_target_network(sess)

        # stop the while loop if game is done
        if done:
            break
    
    file = open('log_episodes.txt', 'a')
    lst = [time.asctime(), episode_counter, global_step_counter, episode_step_counter, episode_reward, episode_max_q_value]
    print(lst)
    file.write(record(lst))
    file.close()

    if episode_counter%1 == 0:
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), './checkpoint/model')
