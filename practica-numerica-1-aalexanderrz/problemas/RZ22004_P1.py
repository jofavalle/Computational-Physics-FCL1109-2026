#Andrés Alexander Rivera Zelaya (RZ22004)
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
#define the derivative
h = 1e-3
def deriv(f, x):
	""" Derivada de punto medio"""
	return (f(x + h/2) - f(x - h/2))/h
#built the Rodriguez formula
n = int(input("Introducir el orden del polinomio: "))
def nderiv(f,x,n):
	"""n-ésima derivada """
	if n == 0:
		return f(x)
	else: 
		return nderiv(lambda y: deriv(f,y), x, n-1)
#Plotteo
t = np.linspace(-1, 1, 100)
a = np.zeros(n + 1)
f = lambda x: (x**2 - 1)**n
a = 1/(2**n * factorial(n))
legn = a * nderiv(f, t, n)


plt.plot(t,legn,label=f'n = {n}')
plt.grid()
plt.title(f'$P_{n}$')
plt.savefig("pole3.png")
plt.legend()
plt.show()
