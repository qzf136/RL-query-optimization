from query_env import Query
import numpy as np
import random
import tensorflow as tf


def l2i(list):
    length = len(list)
    a = 0
    for i in range(length):
        a += list[i] * pow(2, length-i-1)
    return a


def get_max_leagal(Q, s):
    size = np.shape(Q)[1]
    m = -1
    rand = []
    legal = []
    for i in range(size):
        if s[i] == 0:
            legal.append(i)
    for i in legal:
        if m == -1 or Q[0,i] > m:
            m = Q[0,i]
    for i in legal:
        if Q[0,i] == m:
            rand.append(i)
    if len(rand) == 0:
        return -1
    index = random.choice(rand)
    return index


def Q_network(f):
    env = Query(f)
    n = env.n
    state_num = pow(2,n)
    tf.reset_default_graph()
    inputs1 = tf.placeholder(shape=[1,n], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([n,n], 0, 0.1))
    Qout = tf.matmul(inputs1, W)
    nextQ = tf.placeholder(shape=[1,n], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    # 训练网络
    init = tf.initialize_all_variables()
    # 设置超参数
    e = 0.1
    num_episodes = 100
    rList = []
    orderList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # 初始化环境，得到第一个状态观测值
            s = env.reset()
            reward = 0
            order = []
            d = False
            j = 0
            # Q网络
            while j < 99:
                j += 1
            # 根据Q网络和贪心算法(有随机行动的可能)选定当前的动作
                allQ = sess.run(Qout, {inputs1:np.mat(s)})
                a = get_max_leagal(allQ, s)
                # 获取新的状态值和奖励值
                s1, r, d = env.step(a)
                # 通过将新的状态值传入网络获取Q'值
                Q1 = sess.run(Qout, feed_dict={inputs1:np.mat(s1)})
                # 获取最大的Q值并选定我们的动作
                maxQ1 = get_max_leagal(Q1, s1)
                targetQ = allQ
                targetQ[0, a] = r + maxQ1
                # 用目标Q值和预测Q值训练网络
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.mat(s), nextQ:targetQ})
                s = s1
                env.update_state(s1)
                reward += r
                order.append(a)
                if d == True:
                    break
            rList.append(reward)
            orderList.append(order)
    index = np.argmax(rList)
    return orderList[index]


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
    o = Q_network(list)
    print(o)
