# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:26:23 2020

@author: Alvaro Guilherme
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from sympy.combinatorics.graycode import GrayCode
from scipy.special import erf

#%%  Setando os bits
M = 4
b = math.log(M,2)

#%% Alinhar bits
a = GrayCode(b)
bits = list(a.generate_gray())
            
#%% Eb e Dmin
ex = 1
eb = ex/b

dt = math.sqrt((12*ex)/(M**2-1)); 
dn = np.arange(-(M-1),0,2)
dp = np.arange(1,M,2)
d = []
for i in range(len(dn)):
    d.append(dn[i])
for i in range(len(dp)):
    d.append(dp[i])
d = np.array(d)*dt/2

print(sum(abs(d)**2)/M)
#%% Vetor ótimo
n = 10**6
               
x = np.random.randint(0, high=M, size=(n,1))
x = (2*x - (M-1))*dt/2
x_b = []
for i in range(len(x)):
    for j in range(len(d)):
        if x[i] == d[j]:
            x_b.append(bits[j])

print(sum(abs(x)**2)/len(x))
#%% EbN0 e SNR
ebn0_db = np.arange(0,21,1)
ebn0 = 10**(ebn0_db/10)
n0 = eb/ebn0
snr = np.zeros(len(n0))
sigma = np.zeros(len(n0))
for i in range(len(n0)):
    sigma[i] = math.sqrt(n0[i]/2)
    snr[i] = ex/(sigma[i]**2)

#%% Vetores e BER
ber = []
# v = np.zeros(n)
for i in range(len(sigma)):
    y_b = []
    bits_e = 0
    v = sigma[i] * np.random.randn(n,1)
    
    y = x + v
   
    for i in range(len(y)):       
        j = np.argmin(abs(y[i] - d))
        y_b.append(bits[j])

    for k in range(len(y)):
        for q in range(int(b)):
            if y_b[k][q] != x_b[k][q]:
                bits_e += 1
    print(bits_e)
    ber.append(bits_e/(n*b))
    
def qfunc(arg):
    return 0.5-0.5*erf(arg/1.414)

pe = (2*(1-(1/M))*qfunc(np.sqrt(3/(M**2-1)*snr)))/b


#%% Gráficos
ax_x = ebn0_db
ax_y = ber
plt.plot(ax_x,ax_y)
ax_x2 = ebn0_db
ax_y2 = pe
plt.plot(ax_x2,ax_y2)
plt.yscale('log')
plt.xlabel('EbN0_dB')
plt.ylabel('BER')
plt.ylim(10**-5,1)
plt.xlim(0,20)
plt.show()
plt.scatter(x,np.zeros(len(x)))
# plt.scatter(y,np.zeros(len(y)))
plt.show()

#%%Teste
# print(np.random.randint(0, high=M, size=(10,1)))
# print(np.argmin(abs(-1. - d)))