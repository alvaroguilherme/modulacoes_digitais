# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:47:06 2020

@author: Alvaro Guilherme
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from sympy.combinatorics.graycode import GrayCode
from scipy.special import erf

#%% Setando os bits
M = 16
n = 10**5
b = math.log(M,2)
cont = math.sqrt(M)
cont2 = int(n/b)

a = GrayCode(int(b/2))
a2 = GrayCode(int(b))
bits = list(a.generate_gray())
bits2 = list(a2.generate_gray())

#%% Eb
ex = 1
eb = ex/b

dmin = math.sqrt((6*ex)/(M-1)); 
dn = np.arange(-(cont-1),0,2)
dp = np.arange(1,cont,2)
d = []
for i in range(len(dn)):
    d.append(dn[i])
for i in range(len(dp)):
    d.append(dp[i])
    
d = np.array(d)*dmin/2
dj = d*1j
d2 = []
aux3 = 1
for i in range(len(d)):
    for j in range(len(dj)):
            d2.append(d[i]+(dj[j]*aux3))
    aux3 *= -1
    
d2 = np.array(d2)

#%% Vetor ótimo bits          
# x = np.random.randint(0, high=cont, size=(cont2,1))
# x = (2*x - (M-1))*dmin/2
# x = (2*x - (cont-1))*dmin/2
# x_b = []
# xj = x*1j
# x2 = []
# for i in range(len(x)):
#     for j in range(len(dj)):
#         x2.append(x[i]+dj[j])
# x2 = np.array(x2)


# for i in range(len(x2)):
#     for j in range(len(d2)):
#         if x2[i] == d2[j]:
#             x_b.append(bits[j])

data = np.random.randint(0, high=2, size=(int((n*b)*2),1))
data_aux = np.reshape(data,(int(n*2),int(b)))

data_pha = []
data_qua = []
for i in range(int(n)):
    aux = ''
    aux2 = ''
    for  j in range(int(b)):
        if j < b/2:
            aux = aux + str(data_aux[i][j])
        else:
            aux2 = aux2 + str(data_aux[i][j])
    data_pha.append(aux)
    data_qua.append(aux2)

x_b = []
for k in range(len(data_pha)):
    x_b.append(data_pha[k] + data_qua[k])

#%% Vetor ótimo símbolos
x_pha_aux = []
x_qua_aux = []
for k in data_pha:
    x_pha_aux.append(bits.index(k))
for k in data_qua:
    x_qua_aux.append(bits.index(k))
    
x_pha_aux = np.array(x_pha_aux)
x_qua_aux = np.array(x_qua_aux)

#cont == sqrt(M)
x_pha = (2*x_pha_aux - (cont-1))*dmin/2
x_qua = (2*x_qua_aux - (cont-1))*dmin/2

x = x_pha + 1j*x_qua

# plt.scatter(np.real(x),np.imag(x))
# plt.show()

# for i in range(len(x)):
#     for j in range(len(d2)):
#         if x[i] == d2[j]:
#             x_b[i] = bits2[j]
            
#%% EbN0 e SNR
ebn0_db = np.arange(0,21,2)
ebn0 = 10**(ebn0_db/10)
n0 = eb/ebn0
snr = np.zeros(len(n0))
sigma = np.zeros(len(n0))
for i in range(len(n0)):
    sigma[i] = math.sqrt(n0[i])
    snr[i] = ex/(sigma[i]**2)

#%% Vetores e BER
ber = []
# v = np.zeros(n)
for i in range(len(sigma)):
    y_b = []
    bits_e = 0
    v = sigma[i] * (np.random.randn(n,1)+1j*np.random.randn(n,1))
    
    y = []
    for j in range(len(v)):
        y.append(x[j] + v[j])
    y = np.array(y)
        
    for q in range(len(y)):       
        l = np.argmin(abs(y[q] - d2))
        y_b.append(bits2[l])

    for k in range(len(y_b)):
        # for q in range(int(b)):
            if y_b[k] != x_b[k]:
                bits_e += 1
    print(bits_e)
    ber.append(bits_e/(n*b))
    
#%% Teórica
def qfunc(arg):
    return 0.5-0.5*erf(arg/1.414)
pe = (2*(1-(1/(2**b)))*qfunc(np.sqrt((3/(M-1))*snr)))/b
    
#%% Gráficos
plt.scatter(np.real(x),np.imag(x))
# plt.scatter(y,np.zeros(len(y)))
plt.show()
ax_x = ebn0_db
ax_y = ber
plt.plot(ax_x,ax_y,'x')
# plt.show()
ax_x2 = ebn0_db
ax_y2 = pe
plt.plot(ax_x2,ax_y2)
plt.yscale('log')
plt.xlabel('EbN0_dB')
plt.ylabel('BER')
plt.ylim(10**-4,1)
plt.xlim(0,20)
plt.show()

#%% Teste
# dj = d*1j
# a = 4+2j
# b = 2
# print(a+b)
# c = distance.euclidean(a,b)
# print(np.real(d2))
# print(np.imag(d2))
# print(np.random.randint(0, high=2, size=(10,1)))
# print(data_aux[0][0])
# a = ''
# a = a + '1'
# a = 1
# b = str(a)
# print(bits.index('1'))
# a = [1,1,1]
# print(a[:1])
# print(np.argmin(abs(y[0] - d2)))
# print(np.linalg.norm(abs(y[0] - d2)))
# print(distance.euclidean(y[0],d2))