# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:32:57 2020

https://github.com/AdrianHsu/breakout-Deep-Q-Network/blob/master/environment.py
"""

import gym
import numpy as np
from atari_wrapper_openai import make_wrap_atari
import torch


class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False):
        if atari_wrapper:
            clip_rewards = not test  # if not test, clip reward, else not clip reward
            self.env = make_wrap_atari(env_name, clip_rewards)
        else:
            self.env = gym.make(env_name)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, seed):

        """
        Control the randomness of the environment
        """

        self.env.seed(seed)

    def reset(self):
        """
        observation: np.array
            stack 4 last frames, shape: (84, 84, 4)
        """
        observation = self.env.reset()
        # observation=np.array(observation).transpose((2,0,1))

        return np.array(
            observation)  # torch.from_numpy(np.array(observation).transpose((2,0,1))).unsqueeze(0).contiguous()
        # np.expand_dims(np.transpose(np.array(observation),(2,0,1)),axis=0)

    def step(self, action):
        """
        observation: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        reward: int
            wrapper clips the reward to {-1, 0, 1} by its sign
            we don't clip the reward when testing
        done: bool
            whether reach the end of the episode?
        """
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)

        return np.array(
            observation), reward, done, info  # torch.from_numpy(np.array(observation).transpose((2,0,1))).unsqueeze(0).contiguous(), torch.tensor([reward]), done, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()
