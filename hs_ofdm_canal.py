# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:30:43 2021

@author: Alvaro Guilherme
"""

import modulacao
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

#%% Importando o canal
arquivo_canal = loadmat('NB_0_500k.mat')
# print(h.keys())
# print(arquivo_canal['h'])
h = np.array(arquivo_canal['h'])
# h'
hl = h/np.linalg.norm(h)
# hl = 1
# Canal ideal
#%%
tipo = "PAM"
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
# xi = np.reshape(x,(subports,colunas))
xi = np.reshape(x,(subports,colunas), order='F')
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
xi_ifft3 = np.reshape(xi_ifft2,(tam,1), order='F')
np.savetxt('x.txt',xi_ifft3)

#%% Adicionando o canal
hl = np.ravel(hl)
xi_ifft3 = np.ravel(xi_ifft3)
xi_h = np.convolve(xi_ifft3, hl)
xi_h = xi_h[:tam].reshape(-1,1)
np.savetxt('x_h.txt',xi_h)

#%% SNR
snr_db = np.arange(0,32.5,2.5)
snr_lin = 10**(snr_db/10)
sigma1 = np.sqrt(np.var(xi_h)/snr_lin)

#%% Adicionando ruído e calculando a BER
ber2 = []
energia2 = []
for i in range(len(sigma1)):
    y_b2 = []
    bits_e = 0
    # Adicionando ruído AWGN
    v = sigma1[i] * np.random.randn(tam,1)
    yi = xi_h+v
    # yi=xi_h
    #Série/Paralelo (S/P)
    # yi2 = np.reshape(yi,(subports*2+cp,colunas))
    yi2 = np.reshape(yi,(subports*2+cp,colunas), order='F')
    # yi21 = np.reshape(yi,(colunas,subports*2+cp)).T
    #Retirando o prefixo cíclico 
    yi3 = yi2[cp:]
    # energia2.append(np.var(xi_h)/np.var(v))
    
    #%% Zeropadding
    hl = hl.reshape(-1,1)
    hl2 = np.zeros((len(yi3),1))
    hl2[:hl.shape[0]] +=  hl
    
    #%% Aplicando a FFT
    Yi = np.fft.fft(yi3, axis=0, norm='ortho')
    # energia = np.var(Yi)
    H = np.fft.fft(hl2, axis=0)
    # energia = np.var(H)
    
    #%% Equalizador
    Yi4 = Yi/H
    # energia = np.var(Yi4)
    #%% Demapeamento
    Yi3 = np.ones((subports,colunas), dtype = 'complex_')
    Yi3[-1] *= (Yi4[0]+Yi4[subports]*1j)
    for k in range(subports-1):
        Yi3[k] *= Yi4[k+1] 
    energia = np.var(Yi3)
    
    #%%
    if tipo == 'PAM':
        Yi3 = np.real(Yi3)
    #Distância Euclidiana e mapeamento em bits
    for j in range(len(Yi3[0])): 
        for i in range(len(Yi3)):
            l = np.argmin(abs(Yi3[i][j] - d))
            y_b2.append(bits[l])
    #BER
    for k in range(len(y_b2)):
        for q in range(int(b)):
            if y_b2[k][q] != x_b[k][q]:
                bits_e += 1
    ber2.append(bits_e/(n*b))   

#%% Gráficos
ax_x = snr_db
ax_y = ber2
plt.plot(ax_x,ax_y,'x')
# plt.plot(10*np.log10(np.abs(H)))
# ax_x2 = ebn0_db
# ax_y2 = pe
# plt.plot(ax_x2,ax_y2)
plt.yscale('log')
plt.xlabel('SNR_db')
plt.ylabel('BER')
plt.ylim(10**-5,1)
plt.xlim(0,30)
plt.show()

plt.plot(10*np.log10(np.abs(H[:128])))
plt.show()

#%% Teste
# a = np.ravel(h)
# a = [[1, 2], [3, 4]]
# a = np.array(a)
# c = np.array((2,2))
# a2 = a/c
# c = np.zeros((3,3))
# c[:a.shape[0],:a.shape[1]] = a
# a = np.pad(a, ((6, 3)), 'constant', constant_values=(0))
# c = np.zeros(3)
# for i in range(3):
#c[i] = np.dot(a[i],1/a[i])