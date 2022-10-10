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
#Quantidade de bits gerados
n = 10**3
#Quantidade de símbolos
M = 16
#Bits por símbolo
b = math.log(M,2)
cont = math.sqrt(M)

#%% Gerando bits no código Gray
a = GrayCode(int(b/2))
a2 = GrayCode(int(b))
bits = list(a.generate_gray())
bits2 = list(a2.generate_gray())

#%% Eb e Dmin
#Energia média unitária
ex = 1
#Energia de bit média
eb = ex/b

#%% Constelação
#Distância entre os símbolos
dmin = math.sqrt((6*ex)/(M-1)); 
#Gerar constelação
d = np.arange(-(cont-1),cont,2)  
d = np.array(d)*dmin/2
dj = d*1j
d2 = []
aux3 = 1
for i in range(len(d)):
    for j in range(len(dj)):
            d2.append(d[i]+(dj[j]*aux3))
    aux3 *= -1
d2 = np.array(d2)

#%% Sequência de bits transmitidos    
#Gerando bits    
data = np.random.randint(0, high=2, size=(int((n*b)),1))
data_aux = np.reshape(data,(int(n),int(b)))
#Separando em fase e quadratura
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
#Juntando no vetor de bits transmitidos
x_b = []
for k in range(len(data_pha)):
    x_b.append(data_pha[k] + data_qua[k])
    
#%% Sequência de símbolos transmitidos
#Recebendo índice do bit correspondente ao código Gray 
x_pha_aux = []
x_qua_aux = []
for k in data_pha:
    x_pha_aux.append(bits.index(k))
for k in data_qua:
    x_qua_aux.append(bits.index(k))
    
x_pha_aux = np.array(x_pha_aux)
x_qua_aux = np.array(x_qua_aux)

#Mapeando os bits em símbolos
x_pha = (2*x_pha_aux - (cont-1))*dmin/2
x_qua = (2*x_qua_aux - (cont-1))*dmin/2
x = x_pha + 1j*x_qua
            
#%% EbN0 e SNR
ebn0_db = np.arange(0,21,2)
ebn0 = 10**(ebn0_db/10)
n0 = eb/ebn0
sigma = np.sqrt(n0)
snr = ex/(sigma**2)

#%% Vetor Sinal recebido+Ruído e BER
ber = []
for i in range(len(sigma)):
    y_b = []
    bits_e = 0
    #Sequência de ruído AWGN
    v = (sigma[i]/np.sqrt(2)) * (np.random.randn(n,1)+1j*np.random.randn(n,1))
    #Sequência de símbolos recebidos (Transmitidos+Ruído)
    y = []
    for j in range(len(v)):
        y.append(x[j] + v[j])
    y = np.array(y)
    #Distância Euclidiana   
    for q in range(len(y)):       
        l = np.argmin(abs(y[q] - d2))
        y_b.append(bits2[l])
    #BER
    for k in range(len(y_b)):
        for q in range(int(b)):
            if y_b[k][q] != x_b[k][q]:
                bits_e += 1           
    print(bits_e)
    ber.append(bits_e/(n*b))
    
#%% Probabilidade de erro de bit
def qfunc(arg):
    return 0.5-0.5*erf(arg/1.414)
pe = (4*(1-(1/np.sqrt(M)))*qfunc(np.sqrt((3/(M-1))*snr)))/b

#%% Gráficos
#Símbolos transmitidos
plt.scatter(np.real(x),np.imag(x))
plt.show()
plt.grid()
#BER/EbN0_db
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
