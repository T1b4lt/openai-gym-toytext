import gym
import matplotlib.pyplot as plt
from agents.q_agent import QAgent
from utils import utils

N_EPISODES = 5000
N_STEPS = 100
RENDER = False

env = gym.make('Taxi-v3')

actions_dict = {0: 'south', 1: 'north', 2: 'east',
                3: 'west', 4: 'pickup', 5: 'dropoff'}
hist = {}


for i_episode in range(N_EPISODES):
    state = env.reset()
    penalties = 0
    reward_counter = 0
    if RENDER:
        print("############### Ini Episode", i_episode, "###############")
    for t in range(N_STEPS):
        if RENDER:
            env.render()
            print("Actual State:", state)
        action = env.action_space.sample()
        if RENDER:
            print("Action:", actions_dict[action])
        next_state, reward, done, info = env.step(action)
        reward_counter += reward
        if reward == -10:
            penalties += 1
        if RENDER:
            print("Next State:", next_state, "\n")
        state = next_state
        if done:
            break
    if i_episode % 10 == 0:
        print('Episode: {} Reward: {} Steps Taken: {} Penalties: {} Info: {}'.format(
            i_episode, reward_counter, t+1, penalties, info))
    hist[i_episode] = {'reward': reward_counter,
                       'steps': t+1, 'penalties': penalties}
    if RENDER:
        print("############### End Episode", i_episode, "###############")
print("Average reward:", utils.get_average_reward_last_n(hist, N_EPISODES))
print("Average reward of last 100:", utils.get_average_reward_last_n(hist, 100))
print("Average steps:", utils.get_average_steps_last_n(hist, N_EPISODES))
print("Average steps of last 100:", utils.get_average_steps_last_n(hist, 100))
env.close()
