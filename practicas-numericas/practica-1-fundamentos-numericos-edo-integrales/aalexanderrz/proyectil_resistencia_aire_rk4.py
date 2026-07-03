#Andrés Alexander Rivera Zelaya (RZ22004)
import numpy as np
import matplotlib.pyplot as plt

#Constantes y condiciones iniciales
g = 9.8 #m/s
k = 0.8 #Dimensiones variables
n = np.array([1, 1.5, 2], float)
t0 = 0 #s
tf = 25 #s
N = 1000
h = (tf - t0) / N

#Definición de funciones a utilizar
def norm(x, y):
	"""Función para calcular la norma de un vector bidimensional"""
	return np.sqrt(x**2 + y**2)

def f(t, x, n):
	"""Sistema de ecuaciones diferenciales parciales"""
	w = x[2]
	dwdt = -k * (norm(x[2], x[3])**(n-1)) * x[2]
	z = x[3]
	dzdt = -k * (norm(x[2], x[3])**(n-1)) * x[3] - g
	return np.array([w, z, dwdt, dzdt], float)

def rk4(t, h, r, f, n):
	"""Función Runge-Kutta de orden 4"""
	k1 = h * f(t, r, n)
	k2 = h * f(t + h/2, r + k1/2, n)
	k3 = h * f(t + h/2, r + k2/2, n)
	k4 = h * f(t + h, r + k3, n)
	return r + (k1 + 2*k2 + 2*k3 + k4) / 6

#Creación de arrays vacíos
tpoints = np.arange(t0, tf, h)
#soluciones analíticas
tanalitic = np.linspace(0, 2*12.3/g, 100)
hx = 18.23*tanalitic
hy = 12.3*tanalitic - g*(tanalitic**2)/2
hvx = 18.23
hvy = 12.3 - g*tanalitic
col = ['b','r','g'] #creación lista de nombres de colores para plotteo
fig, ax = plt.subplots(1,2, figsize=(12,5.5)) #Creación de figura para el plotteo de valores
#Aplicación del método RK4
for i in range(3): #Implementación de ciclo for para la evaluación de los 3 casos de valor de potencia
	x = np.array([0, 0, 18.23, 12.3], float) #x, y, vx, vy
	traj = [[], [], [], []] #Creación del contenedor de los vectores estado	
	v = []  #Creación del contenedor de |v|
	tiempo = [] #Creación de tiempo para v en caso de un detenimiento prematuro
	for t in tpoints:
		if (t != 0) and (x[1] < 0): #Pausa de cálculo para posiciones verticales negativas (debajo del suelo)
			break
		else:
			tiempo.append(t)
			for l in range(4):
				traj[l].append(x[l])
			v.append(norm(x[2],x[3]))
			x = rk4(t, h ,x, f, n[i])
	
	ax[0].plot(traj[0],traj[1], label=f'n = {n[i]}',color=col[i]) #Plotteo de vy vs vx
	ax[1].plot(tiempo, v, label=f'n = {n[i]}',color=col[i]) #ploteo de v 

#Ploteo
ax[0].plot(hx,hy,label="Analítica",color='orange')
ax[0].set_xlabel('x(t)')
ax[0].set_ylabel('y(t)')
ax[0].set_title('x(t) vs y(t)')
ax[0].grid()
ax[0].legend()
ax[1].set_xlabel('|v|')
ax[1].plot(tanalitic,norm(hvx,hvy),label="Analítica",color='orange')
ax[1].set_ylabel('t')
ax[1].set_title('|v| vs t')
ax[1].grid()
ax[1].legend()
plt.savefig("Problema4.png")
plt.show()


