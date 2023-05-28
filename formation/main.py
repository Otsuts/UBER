import math
import time
import numpy as np
from socket import *

# TCP communicate settings
host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)


def deal(data):
    '''
    convert data string from UE to numpy
    :param data: str: data received from UE
    :return: states of ship1, and states of ship2
    '''
    data = data.split(',')
    # detection distance of sensors
    detection_dis = float(data[26])
    # ship1
    reach1, block1, overturn1, warning1 = 0, 0, 0, 0
    velocity1 = 0
    distance1 = []
    posture1 = []
    pos1 = data[-2].split(' ')  # absolute position in world
    # split the string to x,y,z position
    position1 = {
        'x': float(pos1[0].split('=')[-1]),
        'y': float(pos1[1].split('=')[-1]),
        'z': float(pos1[2].split('=')[-1]),
    }
    MaxCheckSize1 = float(data[25])  # distance from start point to end point
    # distance from destination
    distance_terminal1 = np.float64(data[20]) / MaxCheckSize1
    # overturn detection
    R1, P1, destination_angle1 = np.float64(data[21]), np.float64(data[22]), np.float64(data[24])
    if destination_angle1 <= -90:
        destination_angle1 += 180
    R1 = (R1 + 180) / (180 + 180)  # normalization
    P1 = (P1 + 180) / (180 + 180)
    overturn1 = any([R1 > (30 + 180) / 360, R1 < (-30 + 180) / 360, P1 > (30 + 180) / 360, P1 < (-30 + 180) / 360])
    posture1.append(R1)
    posture1.append(P1)
    for i in range(0, 28):
        if i < 19:
            data[i] = np.float64(data[i]) / detection_dis  # scaling to 1

            if data[i] == 0:
                data[i] = 1
            if i < 19:  # collision detection
                if 500 / detection_dis < data[i] < 3000 / detection_dis:  # too close
                    warning1 = 1
                elif data[i] < 400 / detection_dis:
                    block1 = 1
            distance1.append(data[i])
        if i == 19:  # velocity
            velocity1 = float(data[i]) / 1000
        elif i == len(data) - 1:
            reach1 = float(data[-1])
            break

    # ship2: same as ship1
    reach2, block2, overturn2, warning2 = 0, 0, 0, 0
    velocity2 = 0
    distance2 = []
    posture2 = []
    pos2 = data[-1].split(' ')
    position2 = {
        'x': float(pos2[0].split('=')[-1]),
        'y': float(pos2[1].split('=')[-1]),
        'z': float(pos2[2].split('=')[-1]),
    }
    MaxCheckSize2 = float(data[25 + 28])
    distance_terminal2 = np.float64(data[20 + 28]) / MaxCheckSize2

    data[21 + 28], data[22 + 28], data[23 + 28], data[24 + 28] = np.float64(data[21 + 28]), np.float64(
        data[22 + 28]), np.float64(
        data[23 + 28]), np.float64(data[24 + 28])
    R2, P2, destination_angle2 = data[21 + 28], data[22 + 28], data[24 + 28]
    if destination_angle2 <= -90:
        destination_angle2 += 180
    R2 = (R2 + 180) / (180 + 180)
    P2 = (P2 + 180) / (180 + 180)
    overturn2 = any([R2 > (30 + 180) / 360, R2 < (-30 + 180) / 360, P2 > (30 + 180) / 360, P2 < (-30 + 180) / 360])
    posture2.append(R2)
    posture2.append(P2)

    for i in range(28, 56):
        if i < 19 + 28:
            data[i] = np.float64(data[i]) / detection_dis

            if data[i] == 0:
                data[i] = 1
            if i < 19 + 28:
                if 500 / detection_dis < data[i] < 3000 / detection_dis:
                    warning2 = 1
                elif data[i] < 400 / detection_dis:
                    block2 = 1
            distance2.append(data[i])
        if i == 19 + 28:
            velocity2 = float(data[i]) / 1000
        elif i == 27 + 28:
            reach2 = float(data[i])
            break

    return distance_terminal1, posture1, distance1, warning1, block1, overturn1, reach1, velocity1, destination_angle1, distance_terminal2, posture2, distance2, warning2, block2, overturn2, reach2, velocity2, destination_angle2, position1, position2


def act(get_tcp_state1, get_tcp_state2):
    '''
    receive the state of ship1 and ship2 and act according to their states
    :param get_tcp_state1: states of ship1
    :param get_tcp_state2: states of ship2
    :return: action : ship1's left thruster, ship1's right thruster, ship2's left thruster, ship2's right thruster
    '''
    left1 = right1 = left2 = right2 = 50000
    for index, dist in enumerate(get_tcp_state1['distance']):
        if index < 9:
            left1 += index * (1 - dist) * 5000
            right1 += -index * (1 - dist) * 5000
        if 10 <= index:
            left1 += -(19 - index) * (1 - dist) * 5000
            right1 += (19 - index) * (1 - dist) * 5000
    left1 -= get_tcp_state1['angle'] * 1000
    right1 += get_tcp_state1['angle'] * 1000

    for index, dist in enumerate(get_tcp_state2['distance']):
        if index < 9:
            left2 += index * (1 - dist) * 5000
            right2 += -index * (1 - dist) * 5000
        if 10 <= index:
            left2 += -(19 - index) * (1 - dist) * 5000
            right2 += (19 - index) * (1 - dist) * 5000
    left2 -= get_tcp_state2['angle'] * 1000
    right2 += get_tcp_state2['angle'] * 1000
    clear1, clear2 = all([x == 1 for x in get_tcp_state1['distance']]), all(
        [x == 1 for x in get_tcp_state2['distance']])
    print(clear1, clear2)
    if clear1 and clear2:
        if get_tcp_state1['position']['y'] - get_tcp_state2['position']['y'] > 400:
            left2 = right2 = 0
        elif get_tcp_state1['position']['y'] - get_tcp_state2['position']['y'] < -400:
            left1 = right1 = 0

    if not clear1 or not clear2:
        if math.sqrt((get_tcp_state1['position']['x'] - get_tcp_state2['position']['x']) ** 2 + (
                get_tcp_state1['position']['y'] - get_tcp_state2['position']['y']) ** 2) <= 5000:
            left1 = right1 = 0
    if left1 > 50000:
        left1 = 50000
    if left1 < -50000:
        left1 = -50000

    if left2 > 50000:
        left2 = 50000
    if left2 < -50000:
        left2 = -50000
    return left1, right1, left2, right2


def TCPcommunication():
    # create TCP sockets
    tcpServerSock = socket(AF_INET, SOCK_STREAM)
    # bind sockets to address
    tcpServerSock.bind(addr)
    # start listening
    tcpServerSock.listen(5)

    while True:
        print('-' * 10 + 'waiting for connection' + '-' * 10)
        # connect from client
        tcpClientSock, addr2 = tcpServerSock.accept()
        print('-' * 10 + f'connected from :{addr2}' + '-' * 10)

        while True:
            # receive data from client
            data = tcpClientSock.recv(bufsiz)
            if not data:
                break
            # replace the end \x00 by ''
            Receive_Data = data.decode().replace('\x00', '')
            distance_terminal1, posture1, distance1, warning1, block1, overturn1, reach1, velocity1, destination_angle1, \
            distance_terminal2, posture2, distance2, warning2, block2, overturn2, reach2, velocity2, destination_angle2, position1, position2 = deal(
                Receive_Data)
            # whether restart
            reset = any([block1, block2, overturn1, overturn2, reach1, reach2])
            # get states of ship1
            get_tcp_state1 = {
                'velocity': velocity1,
                'angle': destination_angle1,
                'distance': distance1,
                'position': position1,
                'warning': warning1,
                'block': block1,
                'reach': reach1
            }
            # get states of ship2
            get_tcp_state2 = {
                'velocity': velocity2,
                'angle': destination_angle2,
                'distance': distance2,
                'position': position2,
                'warning': warning2,
                'block': block2,
                'reach': reach2
            }
            # calculate action and send to client
            left1, right1, left2, right2 = act(get_tcp_state1, get_tcp_state2)
            msg = str(left1) + ',' + str(right1) + ',' + str(left2) + ',' + str(right2) + ',' + str(reset)
            tcpClientSock.send(msg.encode())  # send data
            # waiting for update
            time.sleep(1)
        tcpClientSock.close()
    tcpServerSock.close()


if __name__ == '__main__':
    TCPcommunication()
