# César Elías Peña Aparicio

# PROBLEMA 2

import numpy as np
import integrales as inte
import transformacion_xz as tran
import scipy as sp
from scipy.constants import hbar, c, k

T_array = np.array([3,300, 6000]) #TEMPERATURA

A = 1

wc = 5*10**(13)

def f(w, T):
	beta = 1/(k*T)
	w = np.asarray(w) #w debe ser un array para permitir operaciones con T

	with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
		numerador = (hbar*w**3)*np.exp(-w/wc)
		denominador = np.exp(beta*hbar*w)-1.0
		resultado = numerador / denominador

	# Reemplazamos los valores conflictivos en w=0 (que físicamente la función da 0)
	resultado = np.where(w == 0, 0.0, resultado)
    	# Reemplazamos cualquier infinito o NaN residual por 0
	resultado = np.nan_to_num(resultado, nan=0.0, posinf=0.0, neginf=0.0)

	return resultado

#Hacemos un for para cada T

for T in T_array:

	def f_w(w):
		return f(w, T)
	def integrando_z(z):
		z = np.asarray(z)
		res = np.zeros_like(z, dtype=float)
		mask = (z > 0) & (z < 1)
		z_valid = z[mask]

        	# Es necesario incluir la frecuencia de corte para que el cambio de variable tenga sentido físico
		w_escalado = wc * tran.x_of_z(z_valid)

       		 # 2. Evaluamos la función f_w con el w escalado
		f_evaluada = f_w(w_escalado)

       		 # 3. Multiplicamos por el jacobiano (que también debe escalarse por wc)
		res[mask] = f_evaluada * tran.jacobian(z_valid) * wc

		return res

	#Definimos los puntos
	N_puntos = 10000
	N_monte = 100000
	#Integramos
	trap = inte.trapecio(integrando_z, 0, 1, N_puntos)
	simp = inte.simpson(integrando_z, 0, 1, N_puntos)
	monte =  inte.montecarlo(integrando_z, 0, 1, N_monte)

	#Integramos con scipy
	quad = sp.integrate.quad(integrando_z, 0, 1)

	#Comparamos (PARTE A)+(PARTE B)
	print(f"Para una temperatura T = {T} K considerando la integral caculada con scipy como referencia:")
	print(f"La integral de referencia usando scipy es {quad[0]} con un error estimado de {quad[1]}")
	print(f"La integral por método del trapecio es: {trap} con error relativo de {abs((trap-quad[0])/quad[0])}")
	print(f"La integral por método de simpson es {simp} con error relativo de {abs((simp-quad[0])/quad[0])} ")
	print(f"La integral por método monte carlo es {monte} con error relativo de {abs((monte-quad[0])/quad[0])}")

"""
c) A medida aumenta la temperatura también lo hace la energía y de forma bastante grande, 
pasa del orden de 10^14 a 10^22.

d) El método más estable es el método de simpson porque utiliza parabolas y eso nos permite 
hacer mejores aporximaciones de funciones que contengan curvas. Además el error cometido es proporcional
a h^4, lo que hace que para un tamaño de paso especifico sea el que menor error tiene.
"""