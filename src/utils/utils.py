import numpy as np
import json
from datetime import datetime

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

def generate_report_file(config, hist, qtable):
    """
    Generate a report from the history dict.
    """
    report = {}
    report['config'] = config
    report['report'] = {}
    report['report']['total_average_reward'] = get_average_reward_last_n(hist, config['n_episodes'])
    report['report']['last_10_average_reward'] = get_average_reward_last_n(hist, int(config['n_episodes']*0.1))
    report['report']['total_average_steps'] = get_average_steps_last_n(hist, config['n_episodes'])
    report['report']['last_10_average_steps'] = get_average_steps_last_n(hist, int(config['n_episodes']*0.1))
    report['qtable'] = qtable.tolist()
    report['hist'] = hist

    report_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_report.json'
    json.dump(report, open("../resources/reports/"+report_file_name, "w"), indent=4)
    return