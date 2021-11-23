import gym
import matplotlib.pyplot as plt
from agents.q_agent import QAgent
from utils import utils

N_EPISODES = 1000
N_STEPS = 100
EXPLORATION_RATIO = 0.7
LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.9
RENDER = False
IS_SLIPPERY = False

env = gym.make('FrozenLake-v1', is_slippery=IS_SLIPPERY)

actions_dict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
hist = {}

agent = QAgent(env.observation_space, env.action_space, exploration_ratio=EXPLORATION_RATIO, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)

for i_episode in range(N_EPISODES):
    state = env.reset()
    if RENDER:
        print("############### Ini Episode", i_episode, "###############")
    for t in range(N_STEPS):
        if RENDER:
            env.render()
            print("Actual State:", state)
        action = agent.get_next_step(state)
        if RENDER:
            print("Action:", actions_dict[action])
        next_state, reward, done, info = env.step(action)
        if RENDER:
            print("Next State:", next_state, "\n")
        agent.update_qtable(state, action, reward, next_state)
        state = next_state
        if done:
            if i_episode % 10 == 0:
                print('Episode: {} Reward: {} Steps Taken: {} Info: {}'.format(i_episode, reward, t+1, info))
            hist[i_episode] = {'reward': reward, 'steps': t+1}
            break
    if RENDER:
        print("############### End Episode", i_episode, "###############")
print("Average reward:", utils.get_average_reward_last_n(hist, N_EPISODES))
print("Average reward of last 100:", utils.get_average_reward_last_n(hist, 100))
print("Average steps:", utils.get_average_steps_last_n(hist, N_EPISODES))
print("Average steps of last 100:", utils.get_average_steps_last_n(hist, 100))
print("Q-table:")
print(agent.qtable)
"""
rewards_total = [hist[i]['reward'] for i in hist]
steps_total = [hist[i]['steps'] for i in hist]
plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(len(rewards_total), rewards_total, alpha=0.6, color='green', width=N_EPISODES)
plt.show()
plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(len(steps_total), steps_total, alpha=0.6, color='red', width=N_EPISODES)
plt.show()
"""
env.close()