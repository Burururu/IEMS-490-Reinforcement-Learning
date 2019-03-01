#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 04:05:50 2018

@author: Bruce
"""

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])


pixels_num = 6400

hidden_units = 200
gamma = 0.99
lr = 0.01



with tf.variable_scope('policy'):
    pixels = tf.placeholder(dtype=tf.float32, shape=(None, pixels_num), name='pixels')    
    actions = tf.placeholder(dtype=tf.float32, shape=(None,1), name='actions')
    rewards = tf.placeholder(dtype=tf.float32, shape=(None,1), name='rewards')
    
    hidden = tf.layers.dense(pixels, hidden_units, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    logits = tf.layers.dense(hidden, 1, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
    output = tf.sigmoid(logits, name="sigmoid")
#    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=actions, logits=logits, name="cross_entropy")
#    loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy, name="rewards"))
    loss = tf.losses.log_loss(labels=actions, predictions=output, weights=rewards)

    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

env = gym.make("Pong-v0")
#env = env.unwrapped
#env.seed(123)


def baseline_rewards(input_rewards):
    res = np.zeros_like(input_rewards)
    tmp = 0
    for i in range(len(input_rewards)-1, -1, -1):
        tmp = tmp*gamma + input_rewards[i]
        res[i] = tmp
    mean = np.mean(res)
    std = np.std(res)+1e-6
    return (res-mean)/std

total_rewards = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode_counter in range(1000):
        obs = env.reset()
        done = False
        previous_X = np.zeros((80,80))

        X_list = []
        actions_list = []
        rewards_list = []
        
        # start game
        while not done:
            current_X = preprocess(obs)
            delta_X = (current_X - previous_X).reshape((1, pixels_num))
            previous_X = current_X
            
            probs = sess.run(output, feed_dict={pixels: delta_X})
#            action = 2 if probs[0, 0] > np.random.uniform() else 3
            action = 2 if probs[0, 0] < 0.5 else 3
            obs, reward, done, _ = env.step(action)
#            print(probs[0, 0])
            X_list.append(delta_X[0])
            actions_list.append(action)
            rewards_list.append(reward)
            
            if len(actions_list) == 20000:
                break
        
        # Now the game is finished
        base_rewards = baseline_rewards(rewards_list)
        sess.run([loss, optimizer], feed_dict={pixels: np.vstack(X_list), 
                                               actions: np.vstack(actions_list),
                                               rewards: np.vstack(base_rewards)})
        total_rewards.append(sum(rewards_list))
        print(episode_counter, np.mean(total_rewards), sum(rewards_list))
        file = open('log_p2.txt', 'a')
        file.write(str(episode_counter)+' '+str(sum(rewards_list))+'\n')
        file.close()
        

file = open('log_p2.txt', 'r')
res = []
for line in file.readlines():
    print(line)
    res.append(float(line[:-1].split(' ')[1]))
file.close()

plt.plot(np.array(res))
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('p2.png')
