#Andrés Alexander Rivera Zelaya (RZ22004)
import numpy as np
import matplotlib.pyplot as plt

#parámetros constantes
sig = 10
r = 28
b = 8 / 3
t0 = 0
tf = 50
N = 10000
h = (tf - t0) / N
x0 = np.array([1, 0, 0], float) # x, y, z

#Función parámetro para RK4

def f(t, x):
	"""Sistema de ecuaciones diferenciales parciales de Lorenz"""
	dxdt = sig * (x[1] - x[0])
	dydt = r * x[0] - x[1] - x[0] * x[2]
	dzdt = x[0] * x[1] - b * x[2]
	return np.array([dxdt, dydt, dzdt], float)

#Función RK4

def rk2(t, h, x, f):
	k1 = h * f(t, x)
	k2 = h * f(t + h/2, x + k1/2) 
	return x + (k1 + 2*k2)/3

#Preparación de arrays

tpoints = np.arange(t0, tf, h)
xpoints = []
ypoints = []
zpoints = []

#Implementación de RK4
for t in tpoints:
	xpoints.append(x0[0])
	ypoints.append(x0[1])
	zpoints.append(x0[2])
	x0 = rk2(t, h, x0, f)

#Plotteo
fig, ax = plt.subplots(1,2, figsize = (12,5))
ax[0].plot(tpoints, ypoints, color = 'green')
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t)")
ax[0].set_title("Dependencia temporal de y")
ax[0].grid()
ax[1].plot(xpoints, zpoints, color = 'red')
ax[1].set_xlabel("x(t)")
ax[1].set_ylabel("z(t)")
ax[1].set_title("Proyección xz")
ax[1].grid()
plt.savefig("problema_3.png")
plt.show()
