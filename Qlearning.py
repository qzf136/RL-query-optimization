import numpy as np
import query_env as query
import random


def list2int(list):
    length = len(list) // 2
    lst = [list[2*i+1] for i in range(length)]
    a = 0
    for i in range(length):
        a += lst[i] * pow(2, length-i-1)
    return a


def get_max_leagal(Q, s):
    size = np.shape(Q)[1]
    v = list2int(s)
    list = [s[2*i+1] for i in range(len(s)//2)]
    # print("list = ",list)
    m = -1
    rand = []
    legal = []
    for i in range(size):
        if list[i] == 0:
            legal.append(i)
    for i in legal:
        if m == -1 or Q[v,i] > m:
            m = Q[v,i]
    for i in legal:
        if Q[v,i] == m:
            rand.append(i)
    if len(rand) == 0:
        return -1
    index = random.choice(rand)
    return index


def Qlearning(f):
    env = query.Query(f)
    Q = np.zeros([env.observation_space_size, len(env.action_space)])
    lr = .8
    num_episodes = 1000
    rlist = []
    orderList = []
    for i in range(num_episodes):
        s = env.reset()
        d = False
        j = 0
        reward = 0
        order = []
        while j < 200:
            j += 1
            v = list2int(s)
            a = get_max_leagal(Q, s)
            s1, r, d = env.step(a)
            if d:
                break
            Q[v,a] = r + Q[list2int(s1), get_max_leagal(Q, s1)]
            s = s1
            order.append(a)
            reward += r

        rlist.append(reward)
        orderList.append(order)
    index = np.argmax(rlist)
    o = orderList[index]
    return Q, o


if __name__ == '__main__':
    l1 = []
    l2 = []
    list = []
    f = open('data.txt', 'r')
    for line in f:
        lst = line.strip().split(' ')
        if line.__contains__('v'):
            l1.append(int(lst[2]))
        elif line.__contains__('e'):
            l2.append(float(lst[3]))
    for i in range(len(l2)):
        list.append(l1[i])
        list.append(l2[i])
    list.append(l1[len(l1)-1])
    Q, o = Qlearning(list)
    print(o)
