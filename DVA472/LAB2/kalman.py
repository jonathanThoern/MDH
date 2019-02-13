# EXAMPLE FROM EL2320
# 
# This is an example program to simulate the car example described by the
# equations
# 
# p(k+1) = p(k) + dt * s(k)
# s(k+1) = s(k) + dt * u(k)
#
# where p(k) and s(k) is the position and speed respectively
#
# This can be re-written in state-space form as
# 
# x(k+1) = A * x(k) + B * u(k)
#
# where A = [1 dt 0 1], B = [0 dt], x=[p s]
#
# We also assume that we measure the position, which gives
# y(k) = [1 0] * x(k) = C * x(k)
# 
# We also add noise, i.e.
# 
# x(k+1) = A * x(k) + B * u(k) + G * w(k)
# y(k) = C * x(k) + D * v(k)
# where w(k) is process noise and v(k) is measurement noise.
# 
# We assume that G is eye(2) (identity matrix) and that D is 1
# 
# We assume that w(k) and v(k) are white, zero-mean and Gaussian
# 
# For estimation we use the Kalman Filter
#
#
# This program assumes that dt=0.1s and allows you to play with different
# setting for the noise levels. Try with different values for the noise levels

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as math
from numpy import dot, sum, tile, linalg
import random
import math


##########################
# The system model
dt = 0.1
A = np.matrix([[1, dt], [0, 1]])
B = np.matrix([[0], [dt]])
C = np.matrix([1, 0])

x = np.matrix([1, 0.5])
x = x.getH()

u = 0

# the simulated noise 

wStdP = 0.01    # Noise on simulated position
wStdV = 0.1     # Noise on simulated velocity
vStd = 0.1      # Simulated measurement noise on position

##########################
# The Kalman Filter modeled uncertainties and initial values
xhat = np.matrix([-2, 0])
xhat = xhat.getH()

P = np.identity(2)*1
G = np.matrix([[1,0],[0,1]])
D = 1
R = 100 * np.diag([0.01**2, 0.1**2])
Q = 100*0.1**2

# Iterations in simulation. Change for to while true
n = 300

X = np.zeros((2,n+1))
Xhat = np.zeros((2,n+1))
PP = np.zeros((4,n+1))
KK = np.zeros((2,n))
X[:,[0]] = x
Xhat[:,[0]] = xhat

PP[:,0] = np.reshape(P, (1,4)) # MAY BE WRONG!!! (4,1)

for k in range(0,n):
    
    temp1 = np.matrix([wStdP*np.random.randn(1), wStdV*np.random.randn(1)])
    x = (A * x) + (B * u) + temp1 
    y = C * x + D * vStd*np.random.randn(1,1)
 
    X[:,[k+1]] = x
    xhat = A * xhat + B * u
    Acon = A.getH()  
    Gcon = G.getH() #G is identity, T is the same
    P = A * P * Acon + G * R * Gcon
    
############ complete line 83
    Ccon = C.getH()
    # Dcon = D.getH() # Is scalar, cannot be transposed
    K = P * Ccon * linalg.inv(C * P * Ccon + D * Q * D)
#############
    xhat = xhat + K * (y - C * xhat)
    P = P - K * C * P
    
    Xhat[:,[k+1]] = xhat
    KK[:,[k]] = K
    PP[:,[k+1]] = np.reshape(P,(4,1))
    
    # Plot position and speed
    plt.subplot(2,1,1) # Upper subplot
    plt.plot(X[0,1:(k+1)], color = 'red')
    plt.hold(True)
    plt.plot(Xhat[0,1:(k+1)], color = 'blue')
    plt.plot(X[0,1:(k+1)]-Xhat[0,1:(k+1)], color = 'green')
    plt.title('Position (red: true, blue: est, green: error)')
    
    plt.subplot(2,1,2) # Lower subplot
    plt.plot(X[1,1:(k+1)], color = 'red')
    plt.hold(True)
    plt.plot(Xhat[1,1:(k+1)],color = 'blue')
    plt.plot(X[1,1:(k+1)]-Xhat[1,1:(k+1)],color = 'green')
    plt.title('Speed (red: true, blue: est, green: error)')
    
plt.show() # Show plot after calculations are done


"""
E = X - Xhat
disp(sprintf('Standard deviation of error in position (second half): #fm', std(E(1,round(size(E,2)/2):end))))
disp(sprintf('Standard deviation of error in velocity (second half): #fm/s', std(E(2,round(size(E,2)/2):end))))

figure(2)
title('Estimated error covariance')
subplot(2,1,1),
plot(sqrt(PP(1,:)))
title('sqrt(P(1,1))')
subplot(2,1,2),
plot(sqrt(PP(4,:)))
title('sqrt(P(2,2))')

figure(3)
title('Kalman filter gain coefficients')
subplot(2,1,1),
plot(KK(1,:))
title('K(1)')
subplot(2,1,2),
plot(KK(2,:))
title('K(2)')
"""