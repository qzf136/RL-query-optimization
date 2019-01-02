import numpy as np
from functools import reduce


class Query(object):

    def __init__(self, f):
        self.n = (len(f)-1) // 2
        self.action_space = list(np.arange(0, self.n))
        self.state = [0] * self.n
        self.observation_space_size = pow(2, self.n)
        self.f = np.array(f)

    def reset(self):
        self.state = [0] * len(self.f)
        return self.state

    def step(self, action):
        s = self.state.copy()
        s[2*action] = s[2*action+1] = s[2*action+2] = 1

        temp = np.array(s) - np.array(self.state)
        reward = 0
        if sum(temp) == 3:
            reward = -self.calculate(temp*self.f)
        elif sum(temp) == 2:
            v1 = self.calculate(self.state*self.f)
            v2 = self.calculate(s*self.f)
            t = self.calculate(temp*self.f)
            reward = -(v2-v1) / (t-1)

        if sum(self.state) == len(self.f):
            done = True
        else:
            done = False
        self.state = s
        return s, reward, done


    def calculate(self, arr):
        list = []
        start = -1
        end = -1
        for i in range(len(arr)):
            if arr[i] != 0 and start == -1 and end == -1:
                start = i
            elif i > 0 and arr[i] == 0 and arr[i-1] != 0:
                end = i
                list.append(arr[start:end])
                start = -1
                end = -1
            elif i == len(arr)-1 and arr[i] != 0:
                end = len(arr)
                list.append(arr[start:end])
        s = 0
        for l in list:
            v = reduce(lambda x,y:x*y, l)
            s += v
        return s


