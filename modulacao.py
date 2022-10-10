# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:07:41 2020

@author: Alvaro Guilherme
"""
import numpy as np
import math

#%% Setando os bits
#Quantidade de bits gerados
n = 2048
#Quantidade de símbolos
M = 4
#Tipo de modulação
# tipo = "PAM"
#Energia média
ex = 1
#Bits por símbolo
b = math.log(M,2)
#Gerando bits            
data = np.random.randint(0, high=2, size=(int(n*b),1))
#Mapeamento com ou não Gray Code
gray_code = True #True ou False


def modulacao_func(tipo):
    print(tipo)
    if tipo == "PAM":
        import mpam_func
        x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr =  mpam_func.mpam(n,M,ex,b,data,gray_code)
        return x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr
    
    elif tipo == "QAM":
        import mqam_func
        x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr = mqam_func.mqam(n,M,ex,b,data,gray_code)
        return x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr
    
    elif tipo == "PSK":
        import mpsk_func
        x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr = mpsk_func.mpsk(n,M,ex,b,data,gray_code)
        return x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr
        

# x,y,ber,sigma,bits,d,ebn0_db,pe,x_b,snr = modulacao_func(tipo)
