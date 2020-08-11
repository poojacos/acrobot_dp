"""# **LQR Solution for Acrobot**"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:37:36 2020

@author: poojaconsul
"""
l1, l2, m1, m2, g = 0.5, 1, 8, 8, 10
A = np.array([[0,0,1,0], [0,0,0,1], [g/l1, -(g*m2)/(l1*m1), 0, 0], [-g/l1, g*(l1*m1+l1*m2+l2*m2)/(l1*l2*m1), 0,0]])
B = np.array([[0], [0], [-(l1+l2)/(l1*l1*l2*m1)], [(l1*l1*m1 + m2*(l1+l2)*(l1+l2))/(l1*l1*l2*l2*m1*m2)]])
x0 = np.array([0.015,-0.525,0.2,0.8])
Q = np.identity(4)
R = 1

PSS = solve_continuous_are(A, B, Q, R)
K = (1/R) * B.T @ PSS


def f(x, t, A, B, K):
    vals = A - B @ K
    return vals @ x
    
    
t = np.linspace(0, 5, 20)

y = odeint(f, x0, t, args = (A,B,K))

plt.plot(t, y[:,0], 'r', label='theta1')
plt.plot(t, y[:,1], 'y', label='theta1')
plt.plot(t, y[:,2], 'b', label='ang1')
plt.plot(t, y[:,3], 'g', label='ang2')
plt.xlabel("time")
plt.ylabel("x(t)")
plt.title('inti condition theta1:{} theta2:{} ang1:{} ang2:{}'.format(x0[0], x0[1], x0[2], x0[3]))
plt.legend(loc = 'upper right')

"""**Forming a GIF for Acrobot Motion**"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageio as io

states = y

def display(theta1, theta2):
    
    img    = np.zeros((500,500,3),np.uint8)
    origin = (250, 250)
    
    length = 100
    
    Y = length * np.cos(theta1)
    X = length * np.sin(theta1)


    X = int(250 - X)
    Y = int(250 - Y)
    
    cv.line(img,origin,(X,Y),(255,0,0),5)
    cv.line(img,(200,250),(300,250),(255,0,255),5)
    
    
    
    Y1 = length * np.cos(theta2 + theta1)
    X1 = length * np.sin(theta2 + theta1)
    
    X1 = int(X - X1)
    Y1 = int(Y - Y1)
    
    cv.line(img,(X,Y),(X1,Y1),(0,255,0),5)

    return img

images = []

for i in range(states.shape[0]): 
    images.append(display(states[i,0], states[i,1]))

io.mimsave('acrobot.gif', images, duration = 0.5)
