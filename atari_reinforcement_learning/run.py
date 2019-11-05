import gym

import os
import numpy as np
import random

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras.models import load_model

from duel_ddqn import QLayer, AtariStateProcessing

#
### global params
#

ENV_NAME = 'PongDeterministic-v4'
FILE_NAME = 'ddqn.py'

EPISODES = 10
EPSILON = .005

file_path = './model/{}/final/{}/'.format(FILE_NAME, ENV_NAME) + 'model.h5'



#
### load model
#

dqn = load_model(file_path, custom_objects={'QLayer': QLayer})

print('SUCCESS: model loaded - ', file_path)



#
### run env
#

env = gym.make(ENV_NAME)
action_space = env.action_space.n


for episode in range(EPISODES):

    asp = AtariStateProcessing()
    obs = env.reset()
    state = asp.get_state(obs)
    input_shape = state.shape

    done = False
    total_reward = 0

    # start
    while not done:
        env.render()

        # act
        if (np.random.rand() < EPSILON):
            action = random.randrange(action_space)
        else:
            state = state.reshape((-1,) + input_shape)
            q_values = dqn.predict(state)
            action = np.argmax(q_values[0])

        next_obs, reward, done, _info = env.step(action)

        # clean up
        total_reward += reward
        next_state = asp.get_state(next_obs)
        state = next_state.copy()

    print('EPISODE: {}\t\treward: {}'.format( episode+1, total_reward))