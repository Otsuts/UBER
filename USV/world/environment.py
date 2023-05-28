import time
import numpy as np
from utils.util import deal_main_data



def Reward(difference_distance_terminal, distance_terminal, destination_angle, posture, warning, block, overturn, reach,
           velocity):
    '''
    calculate the reward according to current state
    :param difference_distance_terminal: difference of distance to destination in two consequent sample steps
    :param distance_terminal: current distance to destination
    :param destination_angle: destination angle
    :param posture: vehicle posture
    :param warning: whether getting too close to destination
    :param block: whether blocked by obstacles
    :param overturn: whether overturn
    :param reach: whether reaches destination
    :param velocity: current velocity
    :return: reward->float:total reward, restart->bool:whether game over
    '''
    # whether game over
    restart = 0
    # total reward
    block_reward, overturn_reward, reach_reward, warning_reward, velocity_reward, posture_reward, difference_distance_terminal_reward, distance_terminal_reward = 0, 0, 0, 0, 0, 0, 0, 0
    if block == 1:
        block_reward = -2
        restart = 1
    if overturn == 1:
        overturn_reward = -1
        restart = 1
    if reach == 1:
        reach_reward = 20
        restart = 1
    if warning == 1:
        warning_reward = -0.5
    if velocity < 200 / 1000:
        if -1 / ((velocity + 0.00001) * 10) < -2:
            velocity_reward = -1
    elif velocity >= 200 / 1000:
        velocity_reward = 1

    # reward on the distance to destination
    distance_terminal_reward = 0.1 * (1 / distance_terminal)
    # reward on second order distance to destination
    difference_distance_terminal_reward = difference_distance_terminal * 200
    if difference_distance_terminal_reward > 0.9:
        difference_distance_terminal_reward = 0.9
    # reward on the destination angle
    destination_angle_reward = - abs(destination_angle) * 10
    if destination_angle_reward <= -2:
        destination_angle_reward = -2
    reward = block_reward + overturn_reward + reach_reward + warning_reward + \
             difference_distance_terminal_reward + distance_terminal_reward + destination_angle_reward + \
             velocity_reward

    return reward, restart


def last_100_mean_reward(last_100_reward):
    sum = 0
    for item in last_100_reward:
        sum += item
    return sum / len(last_100_reward)
