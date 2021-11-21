import numpy as np

def get_reward_array(hist_dict):
    """
    Get the reward array from the history dict.
    """
    hist = [val.get('reward') for val in hist_dict.values()]
    return np.array(hist)

def get_steps_array(hist_dict):
    """
    Get the steps array from the history dict.
    """
    hist = [val.get('steps') for val in hist_dict.values()]
    return np.array(hist)

def get_average_reward_last_n(hist_dict, n_episodes):
    """
    Get the average reward of the last n_episodes episodes.
    """
    hist = [val.get('reward') for val in hist_dict.values()]
    return np.mean(hist[-n_episodes:])

def get_average_steps_last_n(hist_dict, n_episodes):
    """
    Get the average steps of the last n_episodes episodes.
    """
    hist = [val.get('steps') for val in hist_dict.values()]
    return np.mean(hist[-n_episodes:])