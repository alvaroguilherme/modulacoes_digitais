# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:39:20 2020

@author: Alvaro Guilherme
"""
import modulacao
import numpy as np
from matplotlib import pyplot as plt

#%%
tipo = "QAM"
#Importando função modulação de acordo com o tipo acima
x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr = modulacao.modulacao_func(tipo)
#Número de amostras
n = modulacao.n
#Bits por símbolo
b = modulacao.b
#Quantidade de subportadoras
subports = 128
#Tamanho do prefixo cíclico
cp = 32
colunas = int(n/subports)

# energia = np.var(x)
#Deixando com subportadoras linhas e amostras/subportadoras colunas
xi = np.reshape(x,(subports,colunas))
#Fazendo a IDFT
xi_ifft = np.fft.ifft(xi,norm='ortho')
# energia2 = np.var(xi_ifft,axis=0)

#%% Adicionando o prefixo cíclico
cont = -cp
xi_ifft2 = np.zeros((subports+cp,colunas), dtype = 'complex_')
for i in range(len(xi_ifft2)):
    for j in range(len(xi[0])):
        xi_ifft2[i][j] = xi_ifft[cont][j]
    cont += 1

#%% Paralelo/Série (P/S)
tam = int(len(xi_ifft2)*len(xi_ifft2[0]))
xi_ifft3 = np.reshape(xi_ifft2,(tam,1))

#%% Adicionando ruído e calculando a BER
ber2 = []
for i in range(len(sigma)):
    y_b2 = []
    bits_e = 0
    # Adicionando ruído AWGN
    v = (sigma[i]/np.sqrt(2)) * (np.random.randn(tam,1)+1j*np.random.randn(tam,1))
    yi = xi_ifft3+v
    #Série/Paralelo (S/P)
    yi2 = np.reshape(yi,(subports+cp,colunas))
    #Retirando o prefixo cíclico 
    yi3 = yi2[cp:]
    #Aplicando a FFT
    yi_fft = np.fft.fft(yi3, norm='ortho')
    # yi_fft2 = np.real(yi_fft)
    #Distância Euclidiana e mapeamento em bits
    for i in range(len(yi_fft)): 
        for j in range(len(yi_fft[0])):
            l = np.argmin(abs(yi_fft[i][j] - d))
            y_b2.append(bits[l])
    #BER
    for k in range(len(y_b2)):
        for q in range(int(b)):
            if y_b2[k][q] != x_b[k][q]:
                bits_e += 1
    ber2.append(bits_e/(n*b))   

#%% Gráficos
ax_x = ebn0_db
ax_y = ber2
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

#%% Teste
# e = np.array([1,2,3])
# e = np.fft.ifft(e)
# e = np.fft.fft(e)
# teste = np.fft.fft(xi_ifft2)