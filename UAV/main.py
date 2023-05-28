import time
import numpy as np
import threading

import torch
from tqdm import tqdm
from socket import *
from argparser import parse_args
from model import DQNAgent

global forward, turn, get_tcp_state, reset
host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)
get_tcp_state = []


def Next_state(action, get_tcp_state):
    global forward
    global turn
    forward, turn = int(action[0]), int(action[1])
    # print(action[0],action[1])
    time.sleep(0.2)
    distance_terminal, distance, warning, block, reach, velocity, destination_angle = deal_main_data(
        get_tcp_state)
    nextstate = np.append(distance, destination_angle)
    nextstate = np.append(nextstate, velocity)
    nextstate = np.append(nextstate, distance_terminal).astype(np.float)
    return torch.from_numpy(
        nextstate), distance_terminal, destination_angle, warning, block, reach, velocity


def Reward(difference_distance_terminal, distance_terminal, destination_angle, warning, block, reach,
           velocity):
    restart = 0
    block_reward = reach_reward = warning_reward = 0.0
    if block == 1:
        block_reward = -2
        restart = 1
    if reach == 1:
        reach_reward = 20
        restart = 1
    if warning == 1:
        warning_reward = -0.5
    velocity_reward = velocity / 2000
    reward = block_reward + reach_reward + warning_reward + velocity_reward

    return reward, restart


def deal_main_data(get_tcp_state):
    distance_terminal, distance, warning, block, reach, velocity, destination_angle = get_tcp_state
    return distance_terminal, distance, warning, block, reach, velocity, destination_angle


def deal(data):
    reach, block, warning = 0, 0, 0
    distance = []

    data = data.split(',')
    detection_dis = float(data[-2])

    # 距离终点距离
    distance_terminal = np.float64(data[-4])

    destination_angle = np.float64(data[-5])
    velocity = np.float64(data[-3])
    for i in range(7):
        data[i] = np.float64(data[i]) / detection_dis  # 归一化
        if data[i] == 0:
            data[i] = 1
        if data[i] < 300 / detection_dis:
            block = 1
        if 300 / detection_dis < data[i] < 500 / detection_dis:
            warning = 1
        distance.append(data[i])
    distance = np.array(distance)
    distance_terminal = np.array(float(distance_terminal))
    velocity = np.array(velocity)

    return distance_terminal, distance, warning, block, reach, velocity, destination_angle


def TCPcommuition():
    # 创建tcp套接字，绑定，监听
    tcpServerSock = socket(AF_INET, SOCK_STREAM)  # 创建TCP Socket
    # AF_INET 服务器之间网络通信
    # socket.SOCK_STREAM 流式socket , for TCP
    tcpServerSock.bind(addr)  # 将套接字绑定到地址,
    # 在AF_INET下,以元组（host,port）的形式表示地址.
    tcpServerSock.listen(5)  # 操作系统可以挂起的最大连接数量，至少为1，大部分为5

    while True:
        global forward, turn, get_tcp_state, reset
        reset = 0
        print('waiting for connection')
        # tcp这里接收到的是客户端的sock对象，后面接受数据时使用socket.recv()
        tcpClientSock, addr2 = tcpServerSock.accept()  # 接受客户的连接
        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，
        # 可以用来接收和发送数据。
        # address是连接客户端的地址。
        print('connected from :', addr2)

        forward = 0
        turn = 0
        # t1 = threading.Thread(target=input_control, name='T1')  # 输入推力控制
        # t1.start()
        while True:
            data = tcpClientSock.recv(bufsiz)  # 接收客户端发来的数据
            if not data:
                break
            # 接收数据
            # time.sleep(0.01)
            # start = time.time()
            Receive_Data = data.decode().replace('\x00', '')

            distance_terminal, distance, warning, block, reach, velocity, destination_angle = deal(
                Receive_Data)
            get_tcp_state = [distance_terminal, distance, warning, block, reach, velocity, destination_angle]
            # 发送数据

            msg = str(forward) + ',' + str(turn) + ',' + str(reset)
            tcpClientSock.send(msg.encode())  # 返回给客户端数据
            # end = time.time()
            # print(end - start)
        tcpClientSock.close()
    tcpServerSock.close()


t = threading.Thread(target=TCPcommuition, name='TCP communicate')
t.start()


def main(args):
    state_dim = 24  # 24维数据
    global forward, turn, get_tcp_state, reset
    # 在虚幻引擎没有开始运行时等待
    while not get_tcp_state:
        pass

    model = DQNAgent(state_dim, args.hidden_dim, 8)

    if args.mode == 'train':
        print('-' * 15 + 'trianing starts' + '-' * 15)
        for iter in tqdm(range(args.epoch)):
            # reset the env and get cur state
            step = 0
            reset = 1
            time.sleep(0.2)  # 给虚幻引擎以充足的时间反映，及时重置
            start_time = time.time()
            reset = 0
            difference_distance_terminal = 0
            distance_terminal, distance, warning, block, reach, velocity, destination_angle = deal_main_data(
                get_tcp_state)
            state = np.append(distance, destination_angle)
            state = np.append(state, velocity)
            state = np.append(state, distance_terminal)
            state = torch.from_numpy(state)
            # state:torch.size([24])

            while step <= 1000:
                action, action_index = model.act(state)  # action是一个2*1的numpy数组,是左右推进器的数据
                # get reward
                reward, restart = Reward(
                    difference_distance_terminal,
                    distance_terminal,
                    destination_angle,
                    warning,
                    block,
                    reach,
                    velocity
                )
                # get next state
                next_state, next_distance_terminal, next_destination_angle, warning, block, reach, next_velocity = Next_state(
                    action, get_tcp_state)
                # put next state into pool
                model.exp_pool(state, next_state, action_index, reward, restart)
                difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                # next
                destination_angle = next_destination_angle
                velocity = next_velocity
                distance_terminal = next_distance_terminal
                q_, loss = model.learn()
                print(q_, loss)
                state = next_state
                reset = restart
                step += 1
                if restart:
                    break
            if (iter+1)%args.log_interval == 0:
                model.save()
    if args.mode == 'test':
        model.load()
        print('-' * 15 + 'testing starts' + '-' * 15)
        reset = 1
        time.sleep(0.2)  # 给虚幻引擎以充足的时间反映，及时重置
        reset = 0
        difference_distance_terminal = 0
        distance_terminal, distance, warning, block, reach, velocity, destination_angle = deal_main_data(
            get_tcp_state)
        state = np.append(distance, destination_angle)
        state = np.append(state, velocity)
        state = np.append(state, distance_terminal)
        state = torch.from_numpy(state)
        while True:
            action, action_index = model.act(state)  # action是一个2*1的numpy数组,是左右推进器的数据
            # get reward
            reward, restart = Reward(
                difference_distance_terminal,
                distance_terminal,
                destination_angle,
                warning,
                block,
                reach,
                velocity
            )
            # get next state
            next_state, next_distance_terminal, next_destination_angle, warning, block, reach, next_velocity = Next_state(
                action, get_tcp_state)
            difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
            # next
            destination_angle = next_destination_angle
            velocity = next_velocity
            distance_terminal = next_distance_terminal
            state = next_state
            reset = restart
            if restart:
                break


if __name__ == '__main__':
    args = parse_args()
    main(args)
