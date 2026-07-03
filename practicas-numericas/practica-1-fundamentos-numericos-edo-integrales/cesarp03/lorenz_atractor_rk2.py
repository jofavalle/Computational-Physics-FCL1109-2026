# César Elías Peña Aparicio

# PROBLEMA 3

import matplotlib.pyplot as plt
import numpy as np

# Definimos constantes
sig = 10
r = 28
b_cte = 8/3

# Definimos la función
def f(s,t):
	x = s[0]
	y = s[1]
	z = s[2]
	fx = sig*(y - x)
	fy = r * x - y - x * z
	fz = x * y - b_cte * z
	return np.array([fx, fy, fz], float)

# Definimos los parametros del método
a = 0.0
b = 50.0
N = 10000
h = (b-a)/N

# Definimos el intervalo temporal y las listas
tpoints = np.arange(a, b, h)
xpoints=[]
ypoints=[]
zpoints=[]

# Definimos las condiciones iniciales
s = np.array([1.0, 0, 0], float)

# Implementamos el método RK2
for t in tpoints:
	xpoints.append(s[0])
	ypoints.append(s[1])
	zpoints.append(s[2])
	k1 = h*f(s,t)
	k2 = h*f(s+0.5*k1, t + 0.5*h)
	s += k2
	
	"""
	Como se pide implementar rk2, k3 y k4 no se utilizan. 
	Además, se ha optado por implementar el Método del Punto Medio para trabajar.
	#k3 = h*f(s+0.5*k2, t+0.5*h)
	#k4 = h*f(s+k3, t+h)
	#s+=(k1+2*k2+2*k3+k4)/6
	"""

# Ploteamos lo solicitado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.plot(tpoints,xpoints, color="red")
ax1.set_title("Solución y")
ax1.set_xlabel("t")
ax1.set_ylabel("y(t)")

ax2.plot(xpoints,zpoints, color="blue")
ax2.set_title("Atractor extraño (z vs. x)")
ax2.set_xlabel("Valores de x")
ax2.set_ylabel("Valores de z")

plt.savefig("y(t y atractor extraño)")
plt.tight_layout()
plt.show()

