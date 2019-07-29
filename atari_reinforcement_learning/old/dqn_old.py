#########################
#
# DQN 
# 
#########################


#
### PARAMS & HYPERPARAMS
#

EPSILON = 0.5 #init only
EPSILON_MIN = 0.02
EPSILON_MAX = 1.0
EPSILON_DECAY_STEPS = 300*1000

EPISODES = 5000
BATCH_SIZE = 48
INPUT_SHAPE = (None, 4, 88, 80)
DISCOUNT = 0.99

LEARN_RATE_INIT = 0.001
current_learn_rate = LEARN_RATE_INIT
LEARN_RATE_MIN =  0.00005
LEARN_RATE_STEP_PERCENT = 100*1000


GLOBAL_STEP = 0
COPY_STEPS = 100
STEPS_TRAIN = 4
START_STEPS = 2000

BUFFER_LEN = 20*1000

OPENAI_ENV = 'Pong-v0'
RENDER = True

LOG_DIR = './logs'
MODEL_DIR = './model/{}/'.format(OPENAI_ENV)
MODEL_PATH = MODEL_DIR + 'model.ckpt'


#
### imports & inits
#

import gym

import tensorflow as tf
tf.reset_default_graph()

import numpy as np
from collections import deque, Counter
import random
import datetime as dt
import os

from preprocess import preprocess_obs


# globals
env = gym.make(OPENAI_ENV)
n_actions = env.action_space.n

# mkdir
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


#
### network & policy definition
#

def q_network(X, name_scope):
    '''
    input: (88, 80, 1) array
    3 convolutional layers
    1 fully connected hidden layer
    softmax output
    output: action space (9)
    '''

    # init layers
    initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope(name_scope) as scope:
        # convolutions
        c1 = tf.contrib.layers.conv2d(X, 
                            num_outputs=32,
                            kernel_size=(8,8),
                            stride=4,
                            padding='SAME',
                            weights_initializer=initializer)
        tf.summary.histogram('convolution 1', c1)

        c2 = tf.contrib.layers.conv2d(c1,
                            num_outputs=64,
                            kernel_size=(4,4),
                            stride=2,
                            padding='SAME',
                            weights_initializer=initializer)
        tf.summary.histogram('convolution 2', c2)

        c3 = tf.contrib.layers.conv2d(c2,
                            num_outputs=64,
                            kernel_size=(3,3),
                            stride=1,
                            padding='SAME',
                            weights_initializer=initializer)
        tf.summary.histogram('convolution 3', c3)

        # flatten
        flat = tf.contrib.layers.flatten(c3)

        # fully connected hidden layer
        fc = tf.contrib.layers.fully_connected(flat,
                            num_outputs=128,
                            activation_fn=tf.nn.relu,
                            weights_initializer=initializer)
        tf.summary.histogram('fully connected', fc)

        # output layer
        output = tf.contrib.layers.fully_connected(fc,
                            num_outputs=n_actions,
                            activation_fn=None,
                            weights_initializer=initializer)
        tf.summary.histogram('output', output)

        # store params & weights of network
        vars = {v.name[len(scope.name):] : v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}

        return vars, output


    
def policy_epsilon_greedy(action, step):
    epsilon = max(EPSILON_MIN, 
                EPSILON_MAX - ((step / EPSILON_DECAY_STEPS) * (EPSILON_MAX - EPSILON_MIN)))
    
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return action



def sample_memories(batch_size, exp_buffer):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]

    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]


# experience buffer
exp_buffer = deque(maxlen=BUFFER_LEN)


# set placeholders & initialize q networks
X = tf.placeholder(tf.float32, shape=INPUT_SHAPE)
y = tf.placeholder(tf.float32, shape=(None, 1))
in_training_mode = tf.placeholder(tf.bool)

q_main, q_main_outputs = q_network(X, 'q_main')
q_target, q_target_outputs = q_network(X, 'q_target')

X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(q_target_outputs * tf.one_hot(X_action, n_actions),
                    axis=-1,
                    keep_dims=True)

# copy main q to target q
copy_op = [tf.assign(main_name, q_target[var_name]) for var_name, main_name in q_main.items()]
copy_target_to_main = tf.group(*copy_op)

# cost (diff between actual & predicted)
loss = tf.reduce_mean(tf.square(y - Q_action)) 

# optimizer
learn = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer(learn).minimize(loss)

# log
loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

# checkpoint model
saver = tf.train.Saver()


# run session & episodes

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()
    t0 = dt.datetime.now()

    # episode
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        episodic_reward = 0
        episodic_loss = []
        actions_counter = Counter()
        state = [np.zeros(INPUT_SHAPE[-2:]) for _ in range(4)]

        # for i in range(4):
        #     if RENDER:
        #         env.render()

        #     frame = preprocess_obs(obs)


        while not done:
            # if (episode+1 % 25 == 0) or (episode == 0):
                # env.render()
            if RENDER:
                env.render()

            # preprocess screen & remember last 4 frames
            frame = preprocess_obs(obs)
            state.pop(0)
            state.append(frame)

            # get q values
            action_values = q_main_outputs.eval(feed_dict={X:[state], in_training_mode:False})

            # select action
            greedy_action = np.argmax(action_values, axis=-1)
            actions_counter[str(greedy_action)] += 1
            action = policy_epsilon_greedy(greedy_action, GLOBAL_STEP)

            # take action & observe
            next_obs, reward, done, _info = env.step(action)

            # store in buffer
            next_state = state[1:] + [preprocess_obs(next_obs)]
            exp_buffer.append([state, action, next_state, reward, done])

            # train q network from buffer after STEPS_TRAIN steps taken
            if (GLOBAL_STEP % STEPS_TRAIN == 0) and (GLOBAL_STEP > START_STEPS):
                # sample buffer
                o_state, o_action, o_next_state, o_reward, o_done = sample_memories(BATCH_SIZE, exp_buffer)

                # states
                o_state = [i for i in o_state]
                o_next_state = [i for i in o_next_state]
                
                # next actions
                o_next_action = q_target_outputs.eval(feed_dict={X:o_next_state, in_training_mode:False})

                # reward
                y_batch = o_reward + (DISCOUNT * np.max(o_next_action, axis=-1) * (1 - o_done))

                # merge & write
                mrg_summary = merge_summary.eval(feed_dict={X:o_state,
                                                y:np.expand_dims(y_batch, axis=-1),
                                                X_action:o_action})
                file_writer.add_summary(mrg_summary, GLOBAL_STEP)

                # train
                current_learn_rate = max(LEARN_RATE_INIT - ((GLOBAL_STEP / LEARN_RATE_STEP_PERCENT) / 100),
                                        LEARN_RATE_MIN)
                train_loss, _ = sess.run([loss, optimizer],
                                        feed_dict={X:o_state,
                                            y:np.expand_dims(y_batch, axis=-1),
                                            X_action:o_action,
                                            learn:current_learn_rate,
                                            in_training_mode:True})
                episodic_loss.append(train_loss)
            
            # copy & save main q to target q at step intervals
            if ((GLOBAL_STEP+1) % COPY_STEPS == 0) and (GLOBAL_STEP > START_STEPS):
                copy_target_to_main.run()
                saver.save(sess, MODEL_PATH)
            
            # intra-step updates
            obs = next_obs
            GLOBAL_STEP += 1
            episodic_reward += reward

        print('EPISODE: {}\tACTIONS: {}\tGLOBAL STEPS: {}\tGLOBAL TIME: {}\tLEARN RATE: {}\tMEAN LOSS: {}\tTOTAL REWARD: {}'.format(
                                episode+1, 
                                sum(actions_counter.values()), 
                                GLOBAL_STEP, 
                                dt.datetime.now() - t0,
                                current_learn_rate,
                                np.mean(episodic_loss), 
                                episodic_reward))