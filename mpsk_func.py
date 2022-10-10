# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:23:36 2020

@author: Alvaro Guilherme
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from sympy.combinatorics.graycode import GrayCode
from scipy.special import erf

#%% Função
def mpsk(n,M,ex,b,data,gray_code):
    
    if gray_code:
        # Gerando bits no código Gray
        a2 = GrayCode(int(b))
        bits2 = list(a2.generate_gray())
    
    #%% Eb e Dmin
    #Energia média unitária
    ex = 1
    #Energia de bit média
    eb = ex/b
    
    #%% Constelação
    #Gerar constelação
    d = np.arange(0,M,1)  
    dr = np.cos(2*np.pi*d/M)*math.sqrt(ex)
    dj = np.sin(2*np.pi*d/M)*math.sqrt(ex)
    
    d2 = dr + 1j*dj
    
    #%% Sequência de bits transmitidos    
    #Gerando bits    
    data_aux = np.reshape(data,(int(n),int(b)))
    #Separando em fase e quadratura
    x_b = []
    for i in range(int(n)):
        aux = ''
        for  j in range(int(b)):
                aux = aux + str(data_aux[i][j])
        x_b.append(aux)
        
    #%% Sequência de símbolos transmitidos
    #Recebendo índice do bit correspondente ao código Gray 
    if gray_code:
        x_aux = []
        for k in x_b:
            x_aux.append(bits2.index(k))
    else:
        bits2 = []
        #Convertendo decimal para bit
        def dec_bin(x):
            return (bin(x)[2:])
        
        def tam(b,bit,bit_i):        
            # Deixando todos os bits do mesmo tamanho
            if int(b) > len(bit):
                aux = int(b) - len(bit)
                bit = bit_i[:aux] + bit
            return bit
        bit_i = "0"
        bit_i *= int(b)
        bits3 = []
        for i in range(M):
            bits = dec_bin(i)
            bits3.append(tam(b,bits,bit_i))   
        x_aux = []
        for k in x_b:
            x_aux.append(bits3.index(k))   
            
    x_aux = np.array(x_aux)
    #Mapeando os bits em símbolos
    x_pha = np.cos(2*np.pi*x_aux/M)
    x_qua = np.sin(2*np.pi*x_aux/M)
    for i in range(len(x_pha)):
        if x_pha[i] < 10e-7 and x_pha[i] > 0:
            x_pha[i] = 0
        if x_qua[i] < 10e-7 and x_qua[i] > 0:
            x_qua[i] = 0
            
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
        v = sigma[i] * (np.random.randn(n,1)+1j*np.random.randn(n,1))/math.sqrt(2)
        #Sequência de símbolos recebidos (Transmitidos+Ruído)
        y = []
        for j in range(len(v)):
            y.append(x[j] + v[j])
        y = np.array(y)
        #Distância Euclidiana   
        if gray_code:
            for q in range(len(y)):       
                l = np.argmin(abs(y[q] - d2))
                y_b.append(bits2[l])
        else:
            for q in range(len(y)):       
                l = np.argmin(abs(y[q] - d2))
                y_b.append(bits3[l])
        #BER
        for k in range(len(y_b)):
            for q in range(int(b)):
                if y_b[k][q] != x_b[k][q]:
                    bits_e += 1           
        ber.append(bits_e/(n*b))
        
    #%% Probabilidade de erro de bit
    def qfunc(arg):
        return 0.5-0.5*erf(arg/1.414)
    # pe = (2*qfunc(dmin/np.sqrt(2*n0)))/b
    pe = (2*qfunc((np.sqrt(2)*np.sin(np.pi/M))/sigma))/b
    
    #%% Gráficos
    #Símbolos transmitidos
    plt.scatter(x_pha,x_qua)
    plt.show()
    # plt.grid()
    #BER/EbN0_db
    ax_x = ebn0_db
    ax_y = ber
    plt.plot(ax_x,ax_y,'x',label='Resultado empírico')
    # plt.show()
    ax_x2 = ebn0_db
    ax_y2 = pe
    plt.plot(ax_x2,ax_y2,label='Curva teórica')
    plt.yscale('log')
    plt.xlabel('EbN0_dB')
    plt.ylabel('BER')
    plt.ylim(10**-4,1)
    plt.xlim(0,20)
    plt.legend()
    plt.show()
    return x,y,ber,sigma,bits2,d2,ebn0_db,pe,x_b,snr