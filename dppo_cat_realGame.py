# -*- coding:UTF-8 -*-
'''
这是一一个DPPO的实现
'''
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
import time
from datetime import datetime
from math import ceil, floor
import traceback
import os

from multiprocessing import Process, Event, Queue, Value, Manager, RLock
from multiprocessing.managers import BaseManager
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import backend
# 环境
# from environment import EnvironmentMpsen
# import gym
from environment_all import EnvironmentMpsen

RANDOMSEED = 1  # random seed
'''
超参数
'''
MAX_EPISODE = 1000  # 最大训练的episode数
MAX_EPISODE_STEP = 5000  # 每一个episode的step数
N_WORKER = 5  # worker的个数
GAMMA = 0.9  # 奖励的折扣
A_LR = 0.001  # actor的learning rate
C_LR = 0.005  # critic的learning rate
MIN_BATCH_SIZE = 64  # 更新ppo用的最小的batch size
A_UPDATE_STEPS = 3  # actor update steps
C_UPDATE_STEPS = 3  # critic update steps
EPSILON = 0.2  # clipped的参数
DETA = 0.01


class PPO(object):
    def __init__(self,
                 action_dim,
                 obervation_dim,
                 is_train=True
                 ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.is_train = is_train
        # build network
        self.critic = self._build_cnet()
        self.actor = self._build_anet('pi')
        self.actor_old = self._build_anet('oldpi')

        self.actor_opt = keras.optimizers.Adam(A_LR)
        self.critic_opt = keras.optimizers.Adam(C_LR)

    # 创建critic: 输入的是obervation 输出的是v值
    def _build_cnet(self):
        #Input
        '''
        in_hero = Input(shape=(9,), name='hero')
        in_quality = Input(shape=(9,), name='quality')
        in_level = Input(shape=(9,), name='level')
        in_stage = Input(shape=(1,), name='stage')
        in_numeric = Input(shape=(4,), name='numeric')
        #Embedding
        emb_hero = Embedding(45, 5, input_length=9)(in_hero) #None*9*5
        emb_hero = Lambda(lambda x: backend.mean(x,axis=1))(emb_hero) #None*5    
        emb_quality = Embedding(45, 5, input_length=9)(in_quality)
        emb_quality = Lambda(lambda x: backend.mean(x,axis=1))(emb_quality)
        emb_level = Embedding(45, 5, input_length=9)(in_level)
        emb_level = Lambda(lambda x: backend.mean(x,axis=1))(emb_level)

        emb_stage = Embedding(102, 1)(in_stage)
        emb_stage = Reshape((1,))(emb_stage)
        
        c_input = Concatenate()([in_hero, in_quality, in_level, in_stage, in_numeric])
        '''
        c_input = Input(shape=(self.observation_dim,), dtype='float32', name='critic_observation')
        c_x = Dense(32, activation='relu')(c_input)
        c_x = Dense(16, activation='relu')(c_x)
        c_v = Dense(1)(c_x)
        #c_model = Model(inputs=[in_hero, in_quality, in_level, in_stage, in_numeric], outputs=c_v)
        c_model = Model(inputs=c_input, outputs=c_v)
        return c_model

    # 创建actor：
    def _build_anet(self, name):
        '''
        in_hero = Input(shape=(9,), name='hero')
        in_quality = Input(shape=(9,), name='quality')
        in_level = Input(shape=(9,), name='level')
        in_stage = Input(shape=(1,), name='stage')
        in_numeric = Input(shape=(4,), name='numeric')
        #Embedding
        emb_hero = Embedding(45, 5, input_length=9)(in_hero) #None*9*5
        emb_hero = Lambda(lambda x: backend.mean(x,axis=1))(emb_hero) #None*5    
        emb_quality = Embedding(45, 5, input_length=9)(in_quality)
        emb_quality = Lambda(lambda x: backend.mean(x,axis=1))(emb_quality)
        emb_level = Embedding(45, 5, input_length=9)(in_level)
        emb_level = Lambda(lambda x: backend.mean(x,axis=1))(emb_level)

        emb_stage = Embedding(102, 1)(in_stage)
        emb_stage = Reshape((1,))(emb_stage) # None*1
        a_input = Concatenate()([in_hero, in_quality, in_level, in_stage, in_numeric])
        '''
        a_input = Input(shape=(self.observation_dim,), dtype='float32', name=name + '_observation')
        a_x = Dense(32, activation='relu')(a_input)
        a_x = Dense(16, activation='relu')(a_x)
        logits = Dense(self.action_dim, name=name + '_a')(a_x)
        #a_model = Model(inputs=[in_hero, in_quality, in_level, in_stage, in_numeric], outputs=logits, name=name)
        a_model = Model(inputs=a_input, outputs=logits, name=name)
        return a_model

    # 更新actor： 输入是observation，action和advantage
    def a_train(self, observation, action, advantage):
        observation = np.array(observation, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        advantage = np.array(advantage, dtype=np.float32)  # td-error

        with tf.GradientTape() as tape:
            # actor
            #logits = self.actor([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])
            logits = self.actor(observation)
            dist = tfp.distributions.Categorical(logits=logits)
            entropy = dist.entropy()
            # old actor
            #logits_old = self.actor_old([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])
            logits_old = self.actor_old(observation)
            dist_old = tfp.distributions.Categorical(logits=logits_old)
            # 计算import sampling
            ratio = dist.prob(action) / (dist_old.prob(action) + 0.001)
            # ppo2的方式计算loss 就是截断
            surr = ratio*advantage
            a_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.0-EPSILON, 1.0+EPSILON)*advantage))- 2*entropy
        a_grad = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor.trainable_weights))


    # 更新actor_old的参数
    def update_oldpi(self):
        network_param = self.actor.get_weights()
        self.actor_old.set_weights(network_param)


    # 更新critic 输入是：累计的奖励 cumulative_reward, observation
    def c_train(self, cumulative_reward, observation):
        cumulative_reward = np.array(cumulative_reward, dtype=np.float32)
        observation = np.array(observation, dtype=np.float32)
        with tf.GradientTape() as tape:
            #advantage = cumulative_reward - self.critic([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])
            advantage = cumulative_reward - self.critic(observation)
            c_loss = tf.reduce_mean(tf.square(advantage))
        c_grad = tape.gradient(c_loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grad, self.critic.trainable_weights))


    # 计算advantage V(s') * gamma + r - V(s)
    def cal_advantage(self, observation, cumulative_reward):
        cumulative_reward = np.array(cumulative_reward, dtype=np.float32)
        #advantage = cumulative_reward - self.critic([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])
        advantage = cumulative_reward - self.critic(observation)
        print('ad',advantage.numpy())
        return advantage.numpy()


    # 更新整个网络
    def update(self):
        # global GLOBAL_UPDATE_COUNTER
        while ns.coord_status:
            try:
                if ns.GLOBAL_EPISODE < MAX_EPISODE:
                    UPDATE_EVENT.wait()
                    print('<TRAIN_LOG> ', 'ppo update', ' time:', datetime.now())

                    self.update_oldpi()

                    data = [QUEUE.get() for i in range(QUEUE.qsize())]  # 从所有的worker那收集数据
                    data = np.vstack(data)
                    observation = data[:, :self.observation_dim].astype(np.float32)
                    action = data[:, self.observation_dim: self.observation_dim + 1].astype(np.float32)
                    reward = data[:, -1:].astype(np.float32)
                    # print('len ob',len(observation))
                    advantage = self.cal_advantage(observation, reward)

                    # ppo2比较简单，直接就进行a_train更新：
                    for _ in range(A_UPDATE_STEPS):
                        self.a_train(observation, action, advantage)
                    for _ in range(C_UPDATE_STEPS):
                        self.c_train(reward, observation)

                    UPDATE_EVENT.clear()  # 更新停止
                    ns.GLOBAL_UPDATE_COUNTER = 0  # reset
                    ROLLING_EVENT.set()

                    # 选择动作：输入是observation， 输出是clipped的action
            except Exception as e:
                print("exception occurred")
                traceback.print_exc()


    # choose action for train
    # 主要是加些限制条件，人为增加选择十连抽和另奖励这两个动作的概率
    def choose_action(self, observation, legal_action):
        observation = np.array(observation)
        observation = observation[np.newaxis, :].astype(np.float32)
        illegal_action = [i for i in range(self.action_dim) if i not in legal_action] #拿到不合法的action list
        
        #logits = self.actor([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])
        logits = self.actor(observation)
        logits = logits.numpy().flatten()
        logits_masked = logits.copy()
        logits_masked[illegal_action] = -99999999999
        dist_masked = tfp.distributions.Categorical(logits=logits_masked)  # 构建类别分布，
        a_masked = dist_masked.sample()  # 根据概率随机出动作
        a_masked = a_masked.numpy()
        print('<TRAIN_LOG> ','action',a_masked, logits_masked[a_masked])
        print(a_masked in legal_action)
        
        '''
        dist = tfp.distributions.Categorical(logits)  # 构建类别分布，
        a = dist.sample()  # 根据概率随机出动作
        a = a.numpy()
        print('<TRAIN_LOG> org','action',a, logits[a],logits_masked[a_masked])
        print(a in legal_action)
        '''
        
        return a_masked


    # 计算v值
    def get_v(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :].astype(np.float32)
        #v = self.critic([observation[:,:9],observation[:,9:18],observation[:,18:27],observation[:,27:28],observation[:,28:]])[0, 0]
        v = self.critic(observation)[0,0]
        return v

    # save模型参数
    def save_model_weights(self):
        self.critic.save_weights('ppo_discrete_critic_multiprocessing_seven.h5')
        self.actor.save_weights('ppo_discrete_actor_multiprocessing_seven.h5')
        self.actor_old.save_weights('ppo_discrete_actor_old_multiprocessing_seven.h5')

    # load模型参数
    def load_model_weights(self):
        self.critic.load_weights('ppo_discrete_critic_multiprocessing_seven.h5')
        self.actor.load_weights('ppo_discrete_actor_multiprocessing_seven.h5')
        self.actor_old.load_weights('ppo_discrete_actor_old_multiprocessing_seven.h5')


# 为了分布式跑的类
class Worker(object):
    def __init__(self, wid, ppo):
        self.wid = wid  # 工号
        self.env = EnvironmentMpsen()  # 创建环境
        # self.env.seed(wid*100+RANDOMSEED)
        self.ppo = ppo
        # print('<TRAIN_LOG> ', 'worker id: ', self.wid)

        # 定义一个worker

    def work(self):
        # global GLOBAL_EPISODE, GLOBAL_RUNNING_REWARD, GLOBAL_UPDATE_COUNTER
        print('<TRAIN_LOG> ', 'worker.work', ' time:', datetime.now())
        while ns.coord_status:
            try:
                observation = self.env.reset()
                if observation == None:  # 如果初始化不会返回observation，就初始化obervation为全0的list
                    observation = [0] * self.env.get_observation_length()
                episode_cumulative_reward = 0
                # 记录数据
                buffer_observation = []
                buffer_action = []
                buffer_reward = []

                t0 = datetime.now()
                for t in range(MAX_EPISODE_STEP):
                    # 检查是否被更新，PPO进程正在工作，如果在工作就等待
                    if not ROLLING_EVENT.is_set():  # 查询进程是否被阻塞，如果在阻塞状态，就证明如果global PPO正在更新。否则就可以继续。
                        ROLLING_EVENT.wait()  # worker进程的等待位置，直到PPO完成更新
                        buffer_observation = []
                        buffer_action = []
                        buffer_reward = []  # 清空buffer

                    # 正常跑游戏，收集数据
                    legal_action = self.env.get_legal_action_space()  # 得到当前可以用的action
                    action = self.ppo.choose_action(observation, legal_action)
                    observation_, reward, done, info = self.env.step(action)
                    print('<TRAIN_LOG> ', ' episode: ', ns.GLOBAL_EPISODE, '/', MAX_EPISODE, '  Worker:', self.wid,' reward:',reward, ' action:', action, ' ob:', observation, 'ob_:',observation_, 'legal_action',legal_action, 'step', t, ' done:', done, 'pid:',os.getpid(), ' time:', datetime.now())

                    buffer_observation.append(observation)
                    buffer_action.append(action)
                    buffer_reward.append(reward)
                    observation = observation_
                    episode_cumulative_reward = episode_cumulative_reward + reward

                    # GLOBAL_UPDATE_COUNTER是每个work的在游戏中进行一步，也就是产生一条数据就会+1.
                    # 当GLOBAL_UPDATE_COUNTER大于batch(64)的时候，就可以进行更新。
                    update_rlock.acquire()
                    ns.GLOBAL_UPDATE_COUNTER += 1
                    update_rlock.release()
                    if (t == MAX_EPISODE_STEP - 1) or (ns.GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE) or done:
                        # 计算每个状态对应的V(s')
                        # 要注意，这里的len(buffer) < GLOBAL_UPDATE_COUNTER。所以数据是每个worker各自计算的。
                        if done:
                            v_s_ = 0
                        else:
                            v_s_ = self.ppo.get_v(observation_)
                        discounted_reward = []
                        for r in buffer_reward[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            discounted_reward.append(v_s_)
                        discounted_reward.reverse()

                        # 堆叠成数据，并保存到公共队列中。
                        b_observation = np.vstack(buffer_observation)
                        b_action = np.vstack(buffer_action)
                        b_reward = np.array(discounted_reward)[:, np.newaxis]
                        buffer_observation = []
                        buffer_action = []
                        buffer_reward = []
                        QUEUE.put(np.hstack((b_observation, b_action, b_reward)))  # 把数据放在queue里
                        # 如果数据足够，就开始更新
                        if ns.GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                            ROLLING_EVENT.clear()  # stop collecting data
                            UPDATE_EVENT.set()  # global PPO update

                        if ns.GLOBAL_EPISODE >= MAX_EPISODE:  # stop training
                            ROLLING_EVENT.set()
                            UPDATE_EVENT.set()
                            ns.coord_status = False  # 停止更新
                            print('<TRAIN_LOG> ', 'coord stop,break loop', ' time:', datetime.now())
                            break
                    if done:
                        break

                # record reward changes, plot later
                if len(ns.GLOBAL_RUNNING_REWARD) == 0:
                    ns.GLOBAL_RUNNING_REWARD.append(episode_cumulative_reward)
                else:
                    ns.GLOBAL_RUNNING_REWARD.append(
                        ns.GLOBAL_RUNNING_REWARD[-1] * 0.9 + episode_cumulative_reward * 0.1)

                episode_rlock.acquire()
                ns.GLOBAL_EPISODE += 1
                episode_rlock.release()
                if ns.GLOBAL_EPISODE % 100 == 0:
                    print('<TRAIN_LOG> ', 'save model in training', ' time:', datetime.now())
                    self.ppo.save_model_weights()
                    print('<TRAIN_LOG> ', 'save model in training done !', ' time:', datetime.now())

                print('<TRAIN_LOG> ', ' episode: ', ns.GLOBAL_EPISODE, '/', MAX_EPISODE, ' Worker:', self.wid,'  episode reward: ', episode_cumulative_reward, '  running time: ', datetime.now() - t0)
            except Exception as e:
                print("exception occurred")
                traceback.print_exc()


if __name__ == '__main__':
    # np.random.seed(RANDOMSEED)
    # tf.random.set_seed(RANDOMSEED)

    # action_dim = 1
    # observation_dim = 3
    t_s = datetime.now()
    env = EnvironmentMpsen()
    _ = env.reset()
    action_dim = env.get_action_space_length()
    observation_dim = env.get_observation_length()

    # 定义两组不同的事件，update 和 rolling
    UPDATE_EVENT = Event()  # ppo更新事件
    ROLLING_EVENT = Event()  # worker收集数据事件
    UPDATE_EVENT.clear()  # not update now，相当于把标志位设置为False ppo事件停止
    ROLLING_EVENT.set()  # start to roll out，相当于把标志位设置为True，并通知所有处于等待阻塞状态的线程恢复运行状态。 worker开始工作
    update_rlock = RLock()
    episode_rlock = RLock()

    ns = Manager().Namespace()
    ns.GLOBAL_UPDATE_COUNTER = 0
    ns.GLOBAL_EPISODE = 0
    ns.GLOBAL_RUNNING_REWARD = []
    ns.coord_status = True

    QUEUE = Manager().Queue()


    class MyManager(BaseManager):
        pass


    MyManager.register('Ppo', PPO)
    manager = MyManager()
    manager.start()
    Ppo = manager.Ppo(action_dim, observation_dim, is_train=True)

    # GLOBAL_PPO = PPO(action_dim, observation_dim, is_train=True) #一个global的ppo
    print('<TRAIN_LOG> ', 'GLOBAL_PPO get', ' time:', datetime.now())

    try:
        process = []
        # 为每个worker创建进程
        workers = [Worker(wid=i, ppo=Ppo) for i in range(N_WORKER)]
        for i, worker in enumerate(workers):
            p = Process(target=worker.work, )  # 创建进程
            p.start()  # 开始进程
            print('<TRAIN_LOG> ', ' worker process: ', i, 'pid:', p.pid, ' time:', datetime.now())
            process.append(p)  # 把进程放进进程列表里，方便管理

        # 把一个全局的PPO更新放到进程列表最后
        p = Process(target=Ppo.update, )
        p.start()
        print('update process', 'pid', p.pid, ' time:', datetime.now())
        p.join()
        process.append(p)

        for p in process:
            p.join()
    except Exception as e:
        print("exception occurred")
        traceback.print_exc()

    print('<TRAIN_LOG> ', 'save model', ' time:', datetime.now())
    Ppo.save_model_weights()  # 保存全局的模型参数
    running_reward = pd.DataFrame({'accu_reward':ns.GLOBAL_RUNNING_REWARD})
    running_reward.to_csv('running_reward.csv', index=False)
    print('<TRAIN_LOG> ', ' done', 'time:', datetime.now() - t_s)
