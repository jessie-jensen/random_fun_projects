import gym

import numpy as np 
import os
from collections import deque
import random
import cv2

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras.layers import Conv2D, Flatten, Dense



#
### global params
#

ENV_NAME = 'PongDeterministic-v4'

DISCOUNT = 0.99
LEARN_RATE = 0.0001

MEM_SIZE = 500*1000
BATCH_SIZE = 16
TARGET_COPY_STEPS = 20000
REPLAY_FREQ = 4
REPLAY_START = 10000
SAVE_STEPS = 100*1000

EPSILON_MAX = 1.0
EPSILON_MIN1 = 0.1
EPSILON_MIN2 = 0.01
EPSILON_LINEAR_DECAY1 = (1.0 / 1000000) #* REPLAY_FREQ
EPSILON_LINEAR_DECAY2 = (1.0 / 20000000) #* REPLAY_FREQ

EPISODES = 5000


# 
### obs pre-processing 
#

class AtariStateProcessing():
    def __init__(self, input_shape=(84,84), frames=4):
        self.state_l = [np.zeros(input_shape) for _ in range(frames)]
    

    def preprocess(self, obs, play_area_shift=8):
        obs_new = cv2.resize(obs, (84,110), interpolation=cv2.INTER_AREA)
        obs_new = cv2.cvtColor(obs_new, cv2.COLOR_RGB2GRAY)
        obs_processed = obs_new[110-84-play_area_shift:110-play_area_shift, :]

        return obs_processed
    

    def get_state(self, obs):
        obs_processed = self.preprocess(obs)
        self.state_l.pop(0)
        self.state_l.append(obs_processed)
        state = np.stack(self.state_l)

        return state
        



#
### q network
#

class QNetwork():
    def __init__(self, action_space, input_shape):
        self.input_shape = input_shape
        self.action_space = action_space

        self.init_model()  


    def init_model(self):
        self.model = keras.models.Sequential()

        self.model.add(Conv2D(16,
                        (8, 8),
                        strides=4,
                        padding='valid',
                        activation='relu',
                        input_shape=self.input_shape,
                        data_format='channels_first'))
        self.model.add(Conv2D(32,
                        (4,4),
                        strides=2,
                        padding='valid',
                        activation='relu',
                        data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(256,
                        activation='relu'))
        self.model.add(Dense(self.action_space,
                        activation='linear'))
        
        self.model.compile(loss='mse',
                        optimizer=keras.optimizers.Adam(lr=LEARN_RATE),
                        metrics=['accuracy'])

        self.model.summary()



#
### buffer
#

class ExperienceBuffer():
    def __init__(self, input_shape):
        self.buffer = deque(maxlen=MEM_SIZE)
        self.input_shape = input_shape
    

    def remember(self, state, action, reward, next_state, done):
        state = state.reshape((-1,) + self.input_shape)
        next_state = next_state.reshape((-1,) + self.input_shape)
        self.buffer.append((state, action, reward, next_state, done))

    def get_minibatch(self):
        minibatch = random.sample(self.buffer, BATCH_SIZE)
        return minibatch



#
### double dqn algo
#

class DDQN():
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space.n

        self.asp = AtariStateProcessing()
        obs = self.env.reset()
        state = self.asp.get_state(obs)
        print(state.shape)
        self.input_shape = state.shape

        self.dqn_main = QNetwork(self.action_space, self.input_shape)
        self.dqn_target = QNetwork(self.action_space, self.input_shape)

        self.experience_buffer = ExperienceBuffer(self.input_shape)

        self.total_steps = 0
        self.epsilon = EPSILON_MAX


    def experience_replay(self, minibatch):
        X = np.empty((0,) + self.input_shape)
        y = np.empty((0, self.action_space))

        for state, action, reward, next_state, done in minibatch:
            if not done:
                # value with TARGET
                q_update = reward + (DISCOUNT * np.amax(self.dqn_target.model.predict(next_state)[0]))
            else:
                q_update = reward

            q_values = self.dqn_main.model.predict(state)
            q_values[0][action] = q_update

            X = np.append(X, state, axis=0)
            y = np.append(y, q_values, axis=0)
        
        # train update minibatch to MAIN
        history = self.dqn_main.model.fit(X, y, verbose=0)

        return history


    def policy_egreedy(self, state, eval_mode=False):
        if (np.random.rand() < self.epsilon) and (not eval_mode):
            action = random.randrange(self.action_space)
        else:
            state = state.reshape((-1,) + self.input_shape)
            q_values = self.dqn_main.model.predict(state)
            action = np.argmax(q_values[0])
        
        return action


    def train(self, render=True):
        self.total_steps = 0

        for episode in range(EPISODES):
            # reset
            self.asp = AtariStateProcessing()
            obs = self.env.reset()
            state = self.asp.get_state(obs)

            done = False
            total_reward = 0

            # start
            while not done:
                if render:
                    self.env.render()

                # act
                action = self.policy_egreedy(state)
                next_obs, reward, done, _info = self.env.step(action)
                next_state = self.asp.get_state(next_obs)

                # store in buffer
                self.experience_buffer.remember(state, action, reward, next_state, done)

                # experience replay
                if (self.total_steps > REPLAY_START) and (self.total_steps % REPLAY_FREQ == 0):
                    minibatch = self.experience_buffer.get_minibatch()
                    history = self.experience_replay(minibatch)
                    
                    # update epsilon
                    epsilon_thres = 1000*1000 * REPLAY_FREQ
                    if self.total_steps <= epsilon_thres:
                        self.epsilon = max(EPSILON_MIN1, self.epsilon - EPSILON_LINEAR_DECAY1)
                    else:
                        self.epsilon = max(EPSILON_MIN2, self.epsilon - EPSILON_LINEAR_DECAY2)
                
                # main to target copy
                if self.total_steps % TARGET_COPY_STEPS == 0:
                    self.dqn_target.model.set_weights(self.dqn_main.model.get_weights())
                    print('*** target updated ***')
                
                # target save
                if (self.total_steps % SAVE_STEPS == 0) and (self.total_steps >= SAVE_STEPS):
                    file_dir = './model/{}/'.format(ENV_NAME)
                    file_path = file_dir + 'model.h5'
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    self.dqn_target.model.save(file_path)
                    print('*** target saved to {} ***'.format(file_path))

                # clean up
                state = next_state.copy()
                self.total_steps += 1
                total_reward += reward

            try:
                loss = history.history['loss'][0]
            except:
                loss = np.nan

            print('EPISODE: {}\tepsilon: {:.4f}\ttotal steps: {}\tloss: {:.4f}\treward: {}'.format(
                episode+1,
                self.epsilon,
                self.total_steps,
                loss,
                total_reward))           



if __name__ == "__main__":
    ddqn = DDQN()
    ddqn.train()