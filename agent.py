# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:48:00 2020

@author: Rahul Verma
"""

import torch
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple
import os
from dqn_model import DQN
import torch.nn.functional as F
from summary_writer import TensorboardSummary
import sys

seed = 12323
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.backends.cudnn.benchmark = True

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def to_float(array):
    return array.to(dtype=torch.float32) / 255.


def to_tensor(nparray):
    return torch.from_numpy(nparray.transpose((2, 0, 1))).unsqueeze(0)


class Agent_DQN_Trainer(object):

    def __init__(self, env, args):

        # Training Parameters
        self.args = args
        self.env = env
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma_reward_decay
        self.n_actions = env.action_space.n
        self.output_logs = args.output_logs
        self.step = 8e6
        self.curr_step = 0
        self.ckpt_path = args.save_dir
        self.epsilon = args.eps_start
        self.eps_end = args.eps_end
        self.target_update = args.update_target
        self.observe_steps = args.observe_steps
        self.explore_steps = args.explore_steps
        self.saver_steps = args.saver_steps
        self.resume = args.resume
        self.writer = TensorboardSummary(self.args.log_dir).create_summary()
        # Model Settings

        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(4, self.n_actions)
        self.target_net = DQN(4, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.cuda:
            self.policy_net.to(self.cuda)
            self.target_net.to(self.cuda)

        self.target_net.eval()
        train_params = self.policy_net.parameters()
        self.optimizer = optim.RMSprop(train_params, self.lr, momentum=0.95, eps=0.01)
        self.memory = ReplayMemory(args.replay_memory_size)

        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            self.epsilon = checkpoint['epsilon']
            self.curr_step = checkpoint['step']

            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['episode']))

    def epsilon_greedy_policy(self, observation, nA, test=False):

        observation = to_float(observation).to(self.cuda)
        # print("size of observation->"+str(sys.getsizeof(observation.storage())))
        sample = random.random()

        if test:
            return self.policy_net(observation).max(1)[1].view(1, 1).item()

        if sample <= self.epsilon:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.cuda, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(observation).max(1)[1].view(1, 1)

        return action

    def optimize_model(self):

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.cuda,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([to_float(s) for s in batch.next_state if s is not None])
        state_batch = torch.cat([to_float(s).to(self.cuda) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.cuda)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()

    def train(self):

        current_loss = 0
        train_rewards = []
        train_episode_len = 0.0
        file_loss = open(self.output_logs, "a")
        file_loss.write("episode,step,epsilon,reward,loss,length\n")
        print("Training Started")
        episode = 0
        loss = 0.0

        while self.curr_step < self.step:
            state = to_tensor(self.env.reset())

            # * State is in torch.uint8 format , convert before passing to model*#
            done = False
            episode_reward = 0.0
            train_loss = 0
            s = 0  # length of episode
            while not done:
                # self.env.env.render()

                action = self.epsilon_greedy_policy(state, self.n_actions)

                new_state, reward, done, _ = self.env.step(action.item())  # new_state torch.uint8 format
                new_state, reward = to_tensor(new_state).to(self.cuda), torch.tensor([reward], device=self.cuda)
                episode_reward += reward
                self.memory.push(state, action, new_state, reward)

                if (self.curr_step > self.observe_steps) and (self.curr_step % self.args.update_current) == 0:
                    loss = self.optimize_model()
                    train_loss += loss

                print('Step: %i,  Episode: %i,  Action: %i,  Reward: %.0f,  Epsilon: %.5f, Loss: %.5f' % (
                    self.curr_step, episode, action.item(), reward.item(), self.epsilon, loss), end='\r')

                if self.curr_step > self.observe_steps and self.curr_step % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    # TO CHECK APPROXIMATELY HOW MUCH GPU MEMORY OUR REPLAY MEMORY IS CONSUMING
                    print(torch.cuda.get_device_name(0))
                    print('Memory Usage:')
                    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

                if self.epsilon > self.args.eps_end and self.curr_step > self.observe_steps:
                    interval = self.args.eps_start - self.args.eps_end
                    self.epsilon -= interval / float(self.args.explore_steps)

                self.curr_step += 1
                state = new_state
                s += 1

                if self.curr_step % self.args.saver_steps == 0 and episode != 0 and self.curr_step != 0:
                    k = {'step': self.curr_step, 'epsilon': self.epsilon, 'episode': episode,
                         'policy_state_dict': self.policy_net.state_dict(),
                         'target_state_dict': self.target_net.state_dict(), 'optimizer': self.optimizer.state_dict()}
                    filename = os.path.join(self.ckpt_path, 'ckpt.pth.tar')
                    torch.save(k, filename)

            episode += 1
            train_rewards.append(episode_reward.item())
            train_episode_len += s

            if episode % self.args.num_eval == 0 and episode != 0:
                current_loss = train_loss
                avg_reward_train = np.mean(train_rewards)
                train_rewards = []
                avg_episode_len_train = train_episode_len / float(self.args.num_eval)
                train_episode_len = 0.0
                file_loss.write(
                    str(episode) + "," + str(self.curr_step) + "," + "{:.4f}".format(
                        self.epsilon) + "," + "{:.2f}".format(
                        avg_reward_train) + "," + "{:.4f}".format(current_loss) + "," + "{:.2f}".format(
                        avg_episode_len_train) + "\n")
                file_loss.flush()
                self.writer.add_scalar('train_loss/episode(avg100)', current_loss, episode)
                self.writer.add_scalar('episode_reward/episode(avg100)', avg_reward_train, episode)
                self.writer.add_scalar('length of episode/episode(avg100)', avg_episode_len_train, episode)

            self.writer.add_scalar('train_loss/episode', train_loss, episode)
            self.writer.add_scalar('episode_reward/episode', episode_reward, episode)
            self.writer.add_scalar('epsilon/num_steps', self.epsilon, self.curr_step)
            self.writer.add_scalar('length of episode/episode', s, episode)

        print("GAME OVER")
        # self.env.env.close()
