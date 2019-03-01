# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:46:59 2018

@author: Bruce
"""

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped
#env.seed(123)

state_size = 4
action_size = 2
hidden_size = 100
gamma = 0.95
lr = 1e-2
n_episode = 1000


def baseline_rewards(input_rewards):
    res = np.zeros_like(input_rewards)
    tmp = 0
    for i in range(len(input_rewards)-1, -1, -1):
        tmp = tmp*gamma + input_rewards[i]
        res[i] = tmp
    mean = np.mean(res)
    std = np.std(res)+1e-6
    return (res-mean)/std

with tf.variable_scope('policy'):
    states = tf.placeholder(tf.float32, [None, state_size], name='states')
    actions = tf.placeholder(tf.float32, [None, action_size], name='actions')
    rewards = tf.placeholder(tf.float32, [None,], name='rewards')
    
    layer1 = tf.contrib.layers.fully_connected(inputs=states, num_outputs=hidden_size, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    layer2 = tf.contrib.layers.fully_connected(inputs=layer1, num_outputs=action_size, activation_fn=tf.nn.softmax, weights_initializer=tf.contrib.layers.xavier_initializer())
#    layer_softmax = tf.nn.softmax(layer2)
    layer_softmax = layer2
    
    log_likelihood = tf.reduce_sum(-tf.log(layer_softmax) * actions)
    loss = tf.reduce_mean(log_likelihood * rewards)
    
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

total_rewards_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode_counter in range(n_episode):
        state = env.reset()
        env.render()
        
        done = False # done is false if the game is not over
        states_list, actions_list, rewards_list = [], [], []
        # start game
        for _ in range(200):
            action_distr = sess.run(layer_softmax, feed_dict={states: state.reshape([1,4])})
#            action = np.random.choice(range(action_distr.shape[1]), p=action_distr.ravel())
            action = np.argmax(action_distr[0])
            state, reward, done, info = env.step(action)
            states_list.append(state)
            action_array = np.zeros(action_size)
            action_array[action] = 1
            actions_list.append(action_array)
            rewards_list.append(reward)
            print(episode_counter, _, action_array, action_distr)
            if done:    
                # now the game is finished
                total_rewards_list.append(len(rewards_list)) 
                sess.run([loss, optimizer], feed_dict={states: np.vstack(np.array(states_list)), 
                                                       actions: np.vstack(np.array(actions_list)),
                                                       rewards: baseline_rewards(rewards_list)})
                if episode_counter % 100 == 0:
                    print(episode_counter, np.mean(total_rewards_list), np.max(total_rewards_list))
                file = open('log_p1.txt', 'a')
                file.write(str(episode_counter)+' '+str(total_rewards_list[-1])+'\n')
                file.close()
                break

plt.plot(total_rewards_list)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('p1.png')
    