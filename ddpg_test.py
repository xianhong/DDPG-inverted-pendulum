import numpy as np
import gym
import tensorflow as tf
import json, sys, os
import random
import time
from gym import wrappers

log_dir= 'tmp'
env_to_use = 'Pendulum-v0'

# hyperparameters


# game parameters
env = gym.make(env_to_use)

# set seeds to 0
env.seed(10)
np.random.seed(10)

np.set_printoptions(threshold=np.nan)

outdir = 'tmp/ddpg-agent-results'
env = wrappers.Monitor(env, outdir, force=True,video_callable=lambda episode_id: True)

graph=tf.Graph()
sess = tf.Session(graph=graph)
with graph.as_default():
    saver = tf.train.import_meta_graph(os.path.join(log_dir, 'pendulum-model.ckpt-800.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    state_ph = tf.get_collection("state_ph")[0]
    actions = tf.get_collection("actions")[0]
    is_training_ph = tf.get_collection("is_training_ph")[0]

#####################################################################################################
total_steps = 0
for ep in range(10):
    total_reward = 0
    steps_in_ep = 0
    # Initial state
    observation = env.reset()

    for t in range(1000):
       env.render()
       # choose action based on deterministic policy
       action_for_state, = sess.run(actions, 
			feed_dict = {state_ph: observation[None], is_training_ph: False})
       # take step
       next_observation, reward, done, _info = env.step(action_for_state)
       total_reward += reward

       observation = next_observation
       total_steps += 1
       steps_in_ep += 1
		
       if done: 
            break
    print('Episode %2i,Reward: %7.3f, Steps: %i'%(ep,total_reward,steps_in_ep))
    time.sleep(0.1)		
env.close()
sess.close()
