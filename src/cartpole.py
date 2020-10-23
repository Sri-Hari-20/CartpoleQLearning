import gym
import os
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


# Paths to respective storing locations
BASE_PATH = os.path.join(os.getcwd(), 'store')
NPY_PATH = os.path.join(BASE_PATH, 'qtable')
VID_PATH = os.path.join(BASE_PATH, 'recording')


# Pushing all the acquired args to this class to continue using it
class Cartpole:
    def __init__(self, args, lr=0.1, discount=1, epsilon=0.1, num_ep=1000, decay=25):
        self.args = args
        self.min_lr = lr
        self.discount = discount
        self.min_epsilon = epsilon
        self.decay = decay
        self.num_ep = num_ep
        self.env = gym.make('CartPole-v0')

        # Since veloctiy is +inf to -inf we restrict the statespace to +1 to -1
        self.upper_bounds = [self.env.observation_space.high[0], 0.5,
                             self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5,
                             self.env.observation_space.low[2], -math.radians(50) / 1.]

        # Temp variables to store ranges used for Q table
        self.buckets = (1, 1, 6, 12, self.env.action_space.n,)

        # Make the Q table
        self.q_table = np.zeros(self.buckets)

    """
        CartPole environment has continuous state values, should 'bin' them to allow efficient
        storage of values.
    """

    def convert_to_discrete(self, obs):
        discretized = list()
        for i in range(len(obs)):
            # Normalization between 0 and 1
            scaling = (obs[i] + abs(self.lower_bounds[i])) / \
                (self.upper_bounds[i] - self.lower_bounds[i])
            # New value which 'binned'
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            # Limit check
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def get_action(self, state):
        # Epsilon greedy
        if np.random.random() < self.epsilon:
            # Exploration
            return self.env.action_space.sample()
        else:
            # Exploitation
            return np.argmax(self.q_table[state])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    """
        Apply the update formula here
    """

    def update_q(self, state, action, reward, new_state):
        self.q_table[state][action] += self.learning_rate * (reward + self.discount * np.max(
            self.q_table[new_state]) - self.q_table[state][action])

    def train(self):
        # Train and store weights once done
        # Episodes iteration
        print("Debug: Inside agent's training function")
        average_reward = 0

        reward_history = []
        episode_history = [i for i in range(1, self.num_ep + 1)]

        for i in tqdm(range(self.num_ep)):
            # print("Info: Episode {}".format(i+1))
            current_state = self.convert_to_discrete(self.env.reset())
            done = False
            self.learning_rate = self.get_learning_rate(i)
            self.epsilon = self.get_epsilon(i)
            episode_reward = 0
            # Per episode play
            while not done:
                action = self.get_action(current_state)
                """
                    Info is misc data. Should not be used for actual algorithms
                """
                obs, reward, done, _ = self.env.step(action)
                new_state = self.convert_to_discrete(obs)

                # print("Debug: obs ", new_state)
                self.update_q(current_state, action, reward, new_state)

                # Update state
                current_state = new_state
                episode_reward += reward
            # print("Info: Total reward {}".format(episode_reward))

            # Incremental average update
            average_reward += (1/(i+1)) * (episode_reward - average_reward)

            # For plotting
            reward_history.append(episode_reward)
        print("Average reward at the end of {} episodes is {}".format(
            self.num_ep, average_reward))
        print("All Done. Storing the current weights.")
        file_name = str(time.time()) + "_" + str(average_reward) + ".npy"
        file_path = os.path.join(NPY_PATH, file_name)
        np.save(file_path, self.q_table)
        print("Weights saved as {}".format(file_name))

        # Plot the reward graph
        plt.plot(episode_history, reward_history)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Total reward per episode')
        plt.show()

    def run(self):
        # Always greedy as we trained
        self.epsilon = 0

        # Load the q table
        file_path = os.path.join(NPY_PATH, self.args.file)
        self.q_table = np.load(file_path)
        if self.args.save:
            # Saving footage
            vid_dir = str(time.time())
            vid_path = os.path.join(VID_PATH, vid_dir)
            self.env = gym.wrappers.Monitor(self.env, vid_path)

        done = False
        current_state = self.convert_to_discrete(self.env.reset())
        t = 0

        while not done:
            self.env.render()
            t += 1
            action = self.get_action(current_state)
            obs, _, done, _ = self.env.step(action)
            new_state = self.convert_to_discrete(obs)
            current_state = new_state

        print("Agent lasted {} steps".format(t))
