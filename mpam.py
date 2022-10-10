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

#%% Setando os bits
#Quantidade de bits gerados
n = 10**3
#Quantidade de símbolos
M = 4
#Bits por símbolo
b = math.log(M,2)

#%% Gerando bits no código Gray
a = GrayCode(b)
bits = list(a.generate_gray())
            
#%% Eb e Dmin
#Energia média unitária
ex = 1
#Energia de bit média
eb = ex/b
#%% Constelação
#Distância entre os símbolos
dmin = math.sqrt((12*ex)/(M**2-1)); 
#Gerar constelação
dn = np.arange(-(M-1),0,2)
dp = np.arange(1,M,2)
d = list(dn)+list(dp)
d = np.array(d)*dmin/2

#%% Sequência de símbolos transmitidos              
x = np.random.randint(0, high=M, size=(n,1))
x = (2*x - (M-1))*dmin/2
x_b = []
for i in range(len(x)):
    for j in range(len(d)):
        if x[i] == d[j]:
            x_b.append(bits[j])

#%% EbN0 e SNR
ebn0_db = np.arange(0,21,1)
ebn0 = 10**(ebn0_db/10)
n0 = eb/ebn0
sigma = np.sqrt(n0/2)
snr = ex/(sigma**2)

#%% Vetor Sinal recebido+Ruído e BER
ber = []
# v = np.zeros(n)
for i in range(len(sigma)):
    y_b = []
    bits_e = 0
    #Sequência de ruído AWGN
    v = sigma[i] * np.random.randn(n,1)
    #Sequência de símbolos recebidos (Transmitidos+Ruído)
    y = x + v
    #Distância Euclidiana   
    for i in range(len(y)):       
        j = np.argmin(abs(y[i] - d))
        y_b.append(bits[j])
    #BER
    for k in range(len(y)):
        for q in range(int(b)):
            if y_b[k][q] != x_b[k][q]:
                bits_e += 1
    print(bits_e)
    ber.append(bits_e/(n*b))

#%% Probabilidade de erro de bit
def qfunc(arg):
    return 0.5-0.5*erf(arg/1.414)

pe = (2*(1-(1/M))*qfunc(np.sqrt(3/(M**2-1)*snr)))/b

#%% Gráficos
#Símbolos transmitidos
plt.scatter(x,np.zeros(len(x)))
#BER/EbN0_db
ax_x = ebn0_db
ax_y = ber
plt.plot(ax_x,ax_y,'x')
ax_x2 = ebn0_db
ax_y2 = pe
plt.plot(ax_x2,ax_y2)
plt.yscale('log')
plt.xlabel('EbN0_dB')
plt.ylabel('BER')
plt.ylim(10**-5,1)
plt.xlim(0,20)
plt.show()
plt.show()
