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
# cont = int(M/2)
# def norm(M,cont):
# # z = []
#     d = []
#     for i in range(M*2):
#        if cont%2 == 1:
#            d.append(cont)
#        cont += 1       
       
#     return d
# d = Symbol('d')
# def norm(cont):
#     d = []
#     d1 = np.zeros(cont)
#     for i in range(cont):
#        if i == 0:
#            d1[i] = 1/2
#        else:
#            d1[i] = 1 + d1[i-1]
           
#     d2 = np.zeros(cont)
#     for i in range(cont):
#        if i == 0:
#            d2[i] = -1/2
#        else:
#            d2[i] = (-1 + d2[i-1])   
#     cont2 = -1
#     for i in range(cont):
#         if cont2 <= cont:
#             d.append(d2[cont2])
#             cont2 += -1            
    
#     for i in range(cont):
#         d.append(d1[i])
            
#     return d

# d = norm(cont)


# # Convertendo decimal para bit
# def dec_bin(x):
#     return (bin(x)[2:])

# def tam(b,bit,bit_i):        
#     # Deixando todos os bits do mesmo tamanho
#     if int(b) > len(bit):
#         aux = int(b) - len(bit)
#         bit = bit_i[:aux] + bit
#     return bit

# def f_bits(b,d):
#     bits = []
#     aux2 = 0
#     bit_i = "0"
#     bit_i *= int(b)
    
#     for i in range(M):
#         a = dec_bin(aux2)
#         a = tam(b,a,bit_i)
#         bits.append(a)
#         aux2 += 1
                                      
#     return bits

# bits = f_bits(b,d)

#%% Alinhar bits
a = GrayCode(b)
bits = list(a.generate_gray())
# aux = []
# for i in range(len(bits)):
#     dif = 0
#     for j in range(len(b)):
#         if i > 0:
#             if bits[i][j] != bits[i-1][j]:
#                 dif += 1
#         if dif != 1:
#             aux.append(bits[i])
#             aux2 = i
            
#%% Eb e Dmin
ex = 1
# d3 = 0
# for i in range(len(d)):
#     d3 += d[i]**2
# d3 /= M
# dt = ex/d3   
# d = np.array(d) * dt
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


#%% Vetor ótimo
n = 10**5
x = np.random.randn(n,1) #Tem que ser distribuição uniforme nos símbolos
x_b = []
for i in range(len(x)):
    for j in range(len(d)):
        if j == 0: 
            if x[i] <= (d[j]+(dt/2)):
                x_b.append(bits[j])
                x[i] = d[j]
        elif j > 0 and j < len(d)-1:
            if x[i] <= (d[j]+(dt/2)) and x[i] > (d[j-1]+(dt/2)):
                x_b.append(bits[j])
                x[i] = d[j]
        else:
            if x[i] > (d[j]-(dt/2)):
                x_b.append(bits[j])
                x[i] = d[j]
                        
# for i in range(len(x)):
#     if x[i] < 0:
#         x[i] = -1
#     else:
#         x[i] = 1

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
    # sigma * np.random.randn(...) + mu
# n0 = 2*(sigma**2)
# ebn0_db = 10*(math.log(eb/n0,10))
# snr = (b*eb)/(n0/2)
    y = x + v
    for i in range(len(y)):
        for  j in range(len(d)):
            if j == 0: 
                if y[i] <= (d[j]+(dt/2)):
                    y_b.append(bits[j])
            elif j > 0 and j < len(d)-1:
                if y[i] <= (d[j]+(dt/2)) and y[i] > (d[j-1]+(dt/2)):
                    y_b.append(bits[j])
            else:
                if y[i] > (d[j]-(dt/2)):
                    y_b.append(bits[j])

    for k in range(len(y)):
        for q in range(int(b)):
            if y_b[k][q] != x_b[k][q]:
                bits_e += 1
    # print(bits_e)
    ber.append(bits_e/n)
    
def qfunc(arg):
    return 0.5-0.5*erf(arg/1.414)


pe = 2*(1-(1/M))*qfunc(np.sqrt(3/(M**2-1)*snr))

#%% Gráficos
ax_x = ebn0_db
ax_y = ber
plt.plot(ax_x,ax_y)
# plt.yscale('log')
# plt.xlabel('EbN0_dB')
# plt.ylabel('BER')
# plt.show()
ax_x2 = ebn0_db
ax_y2 = pe
plt.plot(ax_x2,ax_y2)
plt.yscale('log')
plt.xlabel('EbN0_dB')
plt.ylabel('BER')
plt.ylim(10**-4,1)
plt.show()
plt.scatter(x,np.zeros(len(x)))
# plt.scatter(y,np.zeros(len(y)))
# plt.show()
                     
#%% Teste
# t = 98%2
# print(t)
# t = 2
# t2 = -t
# print(t2)
# s = ["aa","bb","cc","dd","ee"]
# for i in range(5):
#     for j in range(2):
#         print(s[i][j])
# print(bin(4))
# a = bin(4)
# def dec_bin(x):
#     return (bin(x)[2:])
# a = dec_to_bin(4)
# a = "00000"
# b = "00"
# x = len(a) - len(b)
# print(a[:x])
# c = a[:x] + b
# print(c)
# a = dec_bin(5)
# print(a[1])
# d = np.array(d)
# print(sum(((d*dt))**2)/M)
# z = np.random.randn(n,1)
# print(z[1])