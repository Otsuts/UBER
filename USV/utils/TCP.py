import numpy as np


def deal(data):
    '''
    preprosessing data received from UE and make it array-form
    :param data: str:data received from UE, separated by ','
    :return: numpy array state data
    '''
    # reach destination, blocked by a wall, overturn or get too close to a wall
    reach, block, overturn, warning = 0, 0, 0, 0
    # current velocity
    velocity = 0
    # distance of sensors
    distance = []
    # posture of ship
    posture = []

    data = data.split(',')
    # distance between start point and destination, used to normalize
    max_dist = float(data[-3])
    # max detection distance of ship sensor
    detection_dis = float(data[-2])

    # normalize distance
    distance_terminal = np.float64(data[20]) / max_dist
    # detect_overturn
    R, P, destination_angle = np.float64(data[21]), np.float64(data[22]), np.float64(data[24])
    # normalization
    R = (R + 180) / (180 + 180)
    P = (P + 180) / (180 + 180)
    # destination angle
    destination_angle = (destination_angle + 180) / (180 + 180) - 0.5
    overturn = any([R > (30 + 180) / 360, R < (-30 + 180) / 360, P > (30 + 180) / 360, P < (-30 + 180) / 360])
    posture.append(R)
    posture.append(P)
    # distance sensors
    for i in range(19):
        # normalized data
        data[i] = np.float64(data[i]) / detection_dis
        if data[i] == 0:
            data[i] = 1
        # collision detection
        if 500 / detection_dis < data[i] < 3000 / detection_dis:  # 进入警告区域
            warning = 1
        elif data[i] < 400 / detection_dis:
            block = 1
        distance.append(data[i])

    # use 1000 to normalize
    velocity = float(data[19]) / 1000
    # whether reached destination
    reach = float(data[-1])

    distance = np.array(distance)
    # distance to destination
    distance_terminal = np.array(float(distance_terminal))
    posture = np.array(posture)
    velocity = np.array(velocity)

    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle
