# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:44:20 2020

@author: Rahul Verma
"""

import argparse
from environment import Environment
import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(312421)
    for i in range(total_episodes):
        state = agent.to_tensor(env.reset())
        done = False
        episode_reward = 0.0

        # playing one game
        while (not done):
            env.env.render()
            action = agent.epsilon_greedy_policy(state, env.action_space.n, test=True)

            state, reward, done, info = env.step(action)
            state = agent.to_tensor(state)
            episode_reward += reward
        rewards.append(episode_reward)

        print('[ episode ', i, '] upclipped reward :', episode_reward)
    env.env.close()

    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent import Agent_DQN_Trainer
        agent = Agent_DQN_Trainer(env, args)
        agent.train()
        agent.writer.close()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent import Agent_DQN_Trainer
        agent = Agent_DQN_Trainer(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
