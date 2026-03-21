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

#INTEGRALES
"""
MÉTODO DEL TRAPECIO
"""
#Recibe la función f(x), los límites de la integral (a, b) y la cantidad de puntos (n)
def trapecio(f, a, b, n):
	x = np.linspace(a, b, n+1)
	h = (b-a)/n
	y = f(x)
	
	I = h*((0.5)*y[0] + np.sum(y[1:n]) + 0.5*y[n]) #np.sum(y[1:n]) va desde el segundo elemento al "n-1"-iesimo elemento
	return I

"""
MÉTODO DE SIMPSON
"""
#Recibe la función f(x), los límites de la integral (a, b) y la cantidad de puntos (n)
def simpson(f,a, b, n):
	if n % 2 == 1:
		n += 1
	x = np.linspace(a, b, n+1)
	h = (b-a)/n
	y = f(x)

	I = h/3*(y[0] + y[n] + 4*np.sum(y[1:n:2])+2*np.sum(y[2:n-1:2]))	
	return I

"""
MÉTODO MONTE CARLO
"""
#Recibe la función f(x), los límites de la integral (a, b) y la cantidad de puntos (n)
def montecarlo(f, a, b, N):
	x = np.random.uniform(a, b, N)
	I = (b-a)*np.mean(f(x))
	return I

"""
CAMBIO DE VARIABLE, INTEGRALES AL INFINITO
"""
# Transformacion
def x_of_z(z):
	return z/(1.0-z)
# Jacobiano dx/dz
def jacobian(z):
	return 1.0/(1.0 - z)**2
# Nueva funcion transformada en [0,1]
def integrando_z(f, z):
	"""Este integrando se le da como argumento a la función Simpson o Trapecio. 
	Tomar en cuenta que los limties se le van a dar son diferentes (0.001, 0.9999). 
	Incluso (h, 0.9999)"""
	x = x_of_z(z)
	return f(x)*jacobian(z)