import argparse
import yaml
import gym
from agents.q_agent import QAgent
from utils import utils


def main(args):

    f = open(args.configfile, "r")
    config = yaml.load(f, Loader=yaml.FullLoader)

    N_EPISODES = config['n_episodes']
    N_STEPS = config['n_steps']
    IS_SLIPPERY = config['is_slippery']
    EXPLORATION_RATIO = config['exploration_ratio']
    LEARNING_RATE = config['learning_rate']
    DISCOUNT_FACTOR = config['discount_factor']
    RENDER = config['render']

    print("\n################ Parameters ################\n")
    print("N_EPISODES:", N_EPISODES)
    print("N_STEPS:", N_STEPS)
    print("IS_SLIPPERY:", IS_SLIPPERY)
    print("EXPLORATION_RATIO:", EXPLORATION_RATIO)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("DISCOUNT_FACTOR:", DISCOUNT_FACTOR)
    print("RENDER:", RENDER)
    print("\n############################################\n")

    env = gym.make('FrozenLake-v1', is_slippery=IS_SLIPPERY)

    actions_dict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
    hist = {}

    agent = QAgent(env.observation_space, env.action_space, exploration_ratio=EXPLORATION_RATIO,
                   learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)

    print("\n\n############### Ini Training ###############\n")
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
                break
        if i_episode % 10 == 0:
            print('Episode: {} Reward: {} Steps Taken: {} Info: {}'.format(
                i_episode, reward, t+1, info))
        hist[i_episode] = {'reward': reward, 'steps': t+1}
        if RENDER:
            print("############### End Episode", i_episode, "###############")
    print("\n############### End Training ###############\n")
    print("\n\n################## Report ##################\n")
    report = {"average_reward": utils.get_average_reward_last_n(hist, N_EPISODES),
              "average_reward_last_10": utils.get_average_reward_last_n(hist, int(N_EPISODES*0.1)),
              "average_steps": utils.get_average_steps_last_n(hist, N_EPISODES),
              "average_steps_last_10": utils.get_average_steps_last_n(hist, int(N_EPISODES*0.1))
              }
    print("Average reward:", report["average_reward"])
    print("Average reward of last 10%("+str(int(N_EPISODES*0.1))+"):",
          report["average_reward_last_10"])
    print("Average steps:", report["average_steps"])
    print("Average steps of last 10%("+str(int(N_EPISODES*0.1))+"):",
          report["average_steps_last_10"])
    print("\nQ-table:")
    print(agent.qtable)
    print("\n################ End Report ################")
    utils.generate_report_file(config, report, hist, agent.qtable)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test argparse')
    parser.add_argument('-f', '--file', help='agent config file',
                        required=True, type=str, dest='configfile')
    args = parser.parse_args()
    main(args)
