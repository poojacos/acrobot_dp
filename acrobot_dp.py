# -*- coding: utf-8 -*-
"""
# **Dynamic Programming Solution for Acrobot Problem**
"""

import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

class ACROBOT(object):
    def __init__(self, l1, l2, m1, m2, g):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.a = self.m1 * self.l1**2 + self.m2 * self.l1**2
        self.b = self.m2 * self.l2**2
        self.c = self.m2 * self.l1 * self.l2
        self.d = self.g * self.m1 * self.l1 + self.g * self.m2 * self.l1
        self.e = self.g * self.m2 * self.l2
        
    def calc_M(self, theta2):
        m = np.zeros((theta2.shape[0], 2, 2))
        m[:,0,0] = self.a + self.b + 2 * self.c * np.cos(theta2)
        m[:,0,1] = self.b + self.c * np.cos(theta2)
        m[:,1,0] = self.b + self.c * np.cos(theta2)
        m[:,1,1] = self.b
        return m

    def calc_C(self, theta):
        """
        ang : angular velocity
        theta: angles [theta1, theta2]
        """
        m = np.zeros((theta.shape[0],2,2))
        m[:, 0,0] = -self.c * np.sin(theta[:, 1] * theta[:, 3])
        m[:, 0,1] = -self.c * np.sin(theta[:, 1] * (theta[:, 2] + theta[:, 3]))
        m[:, 1,0] = self.c * np.sin(theta[:, 1] * theta[:, 2])
        m[:, 1,1] = 0
        return m

    def calc_G(self, theta):
        m = np.zeros((theta.shape[0], 2))
        sin_t12 = np.sin(theta[:,0] + theta[:,1])
        m[:,0] = -self.d * np.sin(theta[:,0]) - self.e * sin_t12
        m[:,1] = -self.e * sin_t12
        return m

    def calc_f(self, M, C, G, ang):
        lower = np.einsum("ijk, ki->ji", -np.linalg.inv(M), (np.einsum("ijk, ki->ji", C,ang.T) + G.T))
        return np.concatenate((ang, lower.T), axis=1)

    def calc_g(self, M):
        lower = np.einsum("ijk, kl->ji", -np.linalg.inv(M), np.array([0,1]).reshape((2,1)))
        return np.concatenate((np.zeros((M.shape[0],2)), lower.T), axis = 1)

    def init_ang(self):
      return np.sqrt(2 * (self.m1 * self.l1 * self.g + self.m2 * self.g * (self.l1 + self.l2))/(self.m1 * self.l1 + self.m2 * self.l1))

def is_terminal(state,dtheta):
    # if state.all() == 0:
    if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
        print(state)
        return True
    return False

if __name__ == "__main__":
    dang, dtheta = 0.1, 0.05
    acro = ACROBOT(0.5, 1, 8, 8, 10)
    u = np.arange(-1, 1, 0.2)
    graph = nx.Graph()
    delt = 0.05
    ang_max, ang_min = 10, -10
    theta_max, theta_min = 2*np.pi, -2*np.pi
    flag = False
    init_ang = acro.init_ang()
    # curr_state = np.array([[np.pi, 0, 0, 0], [np.pi, 0, 0, 0], [1,1,1,1]]).reshape((3, 4)) ##for testing functions
    # curr_state = np.array([np.pi/2, 0, -init_ang, 0]).reshape((1, 4))
    curr_state = np.array([-1.71989292e-03,  8.05879787e-03,  5.45524611e-03, -2.18381832e-02]).reshape((1, 4)) ##converges in four steps
    # curr_state = np.array([[9.04132081e-02, -2.13683176e-01, -2.61851109e-01,  5.62703551e-01]]).reshape((1, 4))
    curr_state[0, :2]= np.round(curr_state[0, :2]/dtheta, 1) * dtheta
    curr_state[0, 2:] = np.round(curr_state[0, 2:4]/dang, 1) * dang
    init_state = tuple([0] + list(curr_state[0]))
    graph.add_node((0,) + tuple(curr_state[0]))

    for t in range(1, 100):
        print('---------STEP {}-----------'.format(t))
        theta2 = curr_state[:, 1]
        ang = curr_state[:, 2:4]
        theta = curr_state[:, 0:2]
        M = acro.calc_M(theta2)
        C = acro.calc_C(curr_state)
        G = acro.calc_G(theta)
        f = acro.calc_f(M,C,G,ang)
        g = acro.calc_g(M)
        # print('g shape ', g.shape)

        news = []
        for control in u:
            ds = f + g * control
            s = ds * delt + curr_state
            # print(s)
            good_theta1 = np.logical_and(s[:,0] >= theta_min, s[:,0] <= theta_max)
            good_theta2 = np.logical_and(s[:,1] >= theta_min, s[:,1] <= theta_max)
            good_ang1 = np.logical_and(s[:,2] >= ang_min, s[:,2] <= ang_max)
            good_ang2 = np.logical_and(s[:,3] >= ang_min, s[:,3] <= ang_max)
            good_theta = np.logical_and(good_theta1, good_theta2)
            good_ang = np.logical_and(good_ang1, good_ang2)
            good = np.logical_and(good_theta, good_ang)
            s = s[good]

            norms = np.linalg.norm(s, axis = 1)
            # print('norm ', norms)
            min_val = np.argmin(norms)
            # print('minimum value of norm state - ', s[min_val])

            s[:, :2] = np.round(s[:, :2]/dtheta, 1) * dtheta
            s[:, 2:] = np.round(s[:, 2:]/dang, 1) * dang
            news.append(s)

        prev_state = curr_state
        curr_state  = np.vstack(news)
        
        print(curr_state)
        #add all them to the graph with current timestep

        print(curr_state.shape, prev_state.shape)
        for state in curr_state:
            #use some heuristic if possible
            if t == 1:
              graph.add_edge((t, ) + tuple(state), (t-1,) + tuple(prev_state[0]), weight = control)
            
            else:  
              for i in range(prev_state.shape[0]):
                graph.add_edge((t, ) + tuple(state), (t-1,) + tuple(prev_state[i]), weight = control)
            # graph.add_node((t, ) + tuple(state))
            if is_terminal(state,dtheta) == True:
                print((t, ) + tuple(state))
                flag = True
                print('final node found !')
                break

        if flag:
            print('Terminal state found')
            break

#find the path 
print(init_state)
path = nx.algorithms.dijkstra_path(graph,init_state, (4,0.0,0.0,-0.0,0.0))
print(path)

path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'r', label = 't1')
plt.plot(path[:, 0], path[:, 2], 'g', label = 't2')
plt.plot(path[:, 0], path[:, 3], 'b', label = 'a1')
plt.plot(path[:, 0], path[:, 4], 'y', label = 'a2')
plt.ylim(-2,2)
plt.xlabel('time')
plt.title('path to equilibrium using DP')
plt.legend('upper right')
plt.show()
