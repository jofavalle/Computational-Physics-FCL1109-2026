"""Librería Física Común Centroamericana (LFCC)"""

import numpy as np

h=1e-5 #Paso para las derivadas numéricas, definido globalmente para nderiv

def derivada_d(f, x, h):
    '''Derivada hacia delante'''
    return (f(x+h)-f(x))/h 

def derivada_c(f, x, h):
	"""Derivada central"""
	return (f(x + h/2) - f(x - h/2))/h #Derivada central

def nderiv(f, x, n):
	"""n-ésima derivada basada en la derivada central"""
	if n == 0:
		return f(x)
	else: 
		return nderiv(lambda y: derivada_c(f, y, h), x, n-1)