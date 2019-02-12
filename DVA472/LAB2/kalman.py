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
#from drawnow import drawnow

##########################
# The system model
dt = 0.1
A = np.matrix([[1, dt], [0, 1]])
B = np.matrix([[0], [dt]])
C = np.matrix([1, 0])

#x = [1, 0.5]'
x = np.matrix([1, 0.5])

#x = np.matrix.getH([1, 0.5])
x = x.getH()

u = 0

# the simulated noise 

wStdP = 0.01    # Noise on simulated position
wStdV = 0.1     # Noise on simulated velocity
vStd = 0.1      # Simulated measurement noise on position

##########################
# The Kalman Filter modeled uncertainties and initial values

#xhat = [-2, 0]'
xhat = np.matrix([-2, 0])
xhat = xhat.getH()
#P = eye(2)*1
P = np.identity(2)*1

#G = eye(2)
G = np.identity(2)
D = 1
#R = 100*diag([0.01^2, 0.1^2])
R = 100 * np.diag([0.01**2, 0.1**2])
Q = 100*0.1**2

n = 100

# X = zeros(2,n+1)
X = np.zeros((2,n+1))
#Xhat = zeros(2,n+1)
Xhat = np.zeros((2,n+1))
#PP = zeros(4,n+1)
PP = np.zeros((4,n+1))
#KK = zeros(2,n)
KK = np.zeros((2,n))

#X(:,1) = x
X[:,[1]] = x
#Xhat(:,1) = xhat
Xhat[:,[1]] = xhat
#PP(:,1) = reshape(P,4,1)

PP[:,1] = np.reshape(P, (1,4)) # MAY BE WRONG!!! (4,1)

#figure(1)
fig = plt.figure()
#drawnow
fig.show()

#for k = 1:n
for k in range(1,n):
    #x = A * x + B * u + [wStdP*randn(1,1), wStdV*randn(1,1)]
    temp1 = np.matrix([wStdP*np.random.randn(1), wStdV*np.random.randn(1)])
    x = (A * x) + (B * u) + temp1 # + np.matrix([[wStdP*np.random.randn(1,1)], [wStdV*np.random.randn(1,1)]])
    #y = C * x + D * vStd*randn(1,1)
    y = C * x + D * vStd*np.random.randn(1,1)
    #X(:,k+1) = x
    X[:,[k+1]] = x

    xhat = A * xhat + B * u
    #P = A * P * A' + G * R * G'
    P = A * P * np.conj(A).T + G * R * np.conj(G).T
############ complete line 83
    #K = P * C' * inv(C * P * C' + D * Q * D')
    K = P * np.conj(C).T * linalg.inv(C * P * np.conj(C).T + D * Q * np.conj(D).T)
#############
    xhat = xhat + K * (y - C * xhat)
    P = P - K * C * P
    #Xhat(:,k+1) = xhat
    Xhat[:,[k+1]] = xhat
    #KK(:,k) = K
    KK[:,[k]] = K
    #PP(:,k+1) = reshape(P,4,1)
    PP[:,[k+1]] = np.reshape(P,(4,1))

    #clf, subplot(2,1,1), 
    #plot(X(1,1:(k+1)),'r')
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(X[1,1:[k+1]],'r')
    

    #hold on, 
    plt.hold(True)

    #plot(Xhat(1,1:(k+1)),'b')
    plt.plot(Xhat[1,1:[k+1]], 'b')
    #plot(X(1,1:(k+1))-Xhat(1,1:(k+1)),'g')
    plt.plot(X[1,1:[k+1]]-Xhat[1,1:[k+1]],'g')
#     axis([0 n+5 -2 7])
    plt.subtitle('Position (red: true, blue: est, green: error)')
    #legend('true','est','error')

    #subplot(2,1,2), 
    #plot(X(2,1:(k+1)),'r')
    fig.subplot(2,1,2)
    fig.plot(X[2,1:[k+1]],'r')
    #hold on,
    fig.hold(True)
    #plot(Xhat(2,1:(k+1)),'b')
    fig.plot(Xhat[2,1:[k+1]],'b')
    #plot(X(2,1:(k+1))-Xhat(2,1:(k+1)),'g')
    fig.plot(X[2,1:[k+1]]-Xhat[2,1:[k+1]],'g')
#     axis([0 n+5 -5 5])
    fig.subtitle('Speed (red: true, blue: est, green: error)')
    #legend('true','est','error')

    fig.show()
end

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