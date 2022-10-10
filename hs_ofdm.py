# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:38:40 2021

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
#Recebendo o último termo e separando sua parte real e imaginária
l = xi[-1]
x1 = np.real(l)
x2 = np.imag(l)

#%% Criando vetor com mapeamento simétrico hermitiano (Hermitian symmetric mapping)
xi2 = np.ones((subports*2,colunas), dtype='complex_')
xi2[0] *= x1
xi2[subports] *= x2
cont2 = subports+1
for i in range(subports-1):
    xi2[i+1] *= xi[i]
    xi2[cont2] *= xi[-i-2].conjugate()
    cont2 += 1
#%% Fazendo a IDFT
xi_ifft = np.fft.ifft(xi2, axis=0, norm='ortho')
# energia = np.var(xi_ifft)
xi_ifft = np.real(xi_ifft)
# energia2 = np.var(xi_ifft,axis=0)

#%% Adicionando o prefixo cíclico
cont = -cp
xi_ifft2 = np.zeros((subports*2+cp,colunas))
for i in range(len(xi_ifft2)):
    for j in range(len(xi_ifft[0])):
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
    v = (sigma[i]) * (np.random.randn(tam,1))
    yi = xi_ifft3+v
    #Série/Paralelo (S/P)
    yi2 = np.reshape(yi,(subports*2+cp,colunas))
    #Retirando o prefixo cíclico 
    yi3 = yi2[cp:]
    #%% Aplicando a FFT
    yi_fft = np.fft.fft(yi3, axis=0, norm='ortho')
    # energia = np.var(yi_fft)
    #%% De-mapeamento
    yi_fft3 = np.ones((subports,colunas), dtype = 'complex_')
    yi_fft3[-1] *= (yi_fft[0]+yi_fft[subports]*1j)
    for i in range(subports-1):
        yi_fft3[i] *= yi_fft[i+1]
    #%%
    # yi_fft2 = np.real(yi_fft)
    #Distância Euclidiana e mapeamento em bits
    for i in range(len(yi_fft3)): 
        for j in range(len(yi_fft3[0])):
            l = np.argmin(abs(yi_fft3[i][j] - d))
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
# e = np.array(([1,2,3],[1,2,3])) 
# c = np.random.randn(128,8)
# e1 = 2+3j
# e2 = e1.conjugate()
# e = np.fft.ifft(e)
# e = np.fft.fft(c.T, axis=0, norm='ortho')
# print(e[-1])
# teste = np.fft.fft(xi_ifft2)