# César Elías Peña Aparicio
# PROBLEMA 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import math

h = 0.01

#Definimos el binomio dentro del polinomio de legendre:
def funcion(n):
    return lambda x: (x**2 - 1)**n

#Definimos la función para derivar n veces:
def derivada_n(f, orden, h):
    if orden == 0:
        return f
    else:
        derivada_ante = derivada_n(f, orden - 1, h)
        return lambda x: (derivada_ante(x + h) - derivada_ante(x - h)) / (2 * h)

#Definimos la función para calcular el n-esimo polinomio:
def legendre_n(n, x, h):
    f = funcion(n)
    deriv = derivada_n(f, n, h)
    coef = 1 / ((2**n) * math.factorial(n))
    return coef * deriv(x)

#Definimos el dominio de evaluación:
x_vals = np.linspace(-1, 1, 200)

#Dfinimos parametros para las graficas:
fig, axes = plt.subplots(4, 2, figsize=(10, 8))
axes = axes.flatten()

#Utilizamos un for para evaluar desde n = 1, hasta n=8 y encontrar P1(x) hasta P8(x):
for n in range(1,9):
    P_n = legendre_n(n, x_vals, h) #Resultado de nuestro codigo

    P_n_lib = legendre(n) # Resultado analítico usando la librería scipy
    Poli_L = P_n_lib(x_vals)

    #Ploteamos
    # Generación de las gráficas
    ax = axes[n-1]
    ax.plot(x_vals, P_n, label=f'Numérico P_{n}(x)', linestyle='--', linewidth=3, color='blue')
    ax.plot(x_vals, Poli_L, label=f'Analítico P_{n}(x)', alpha=0.7, color='red')
    ax.set_title(f'Polinomio de Legendre orden n={n}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("Problema1")
plt.show()

"""
c)

Se ejecutó el script 2 veces para generar los plots con h=0.01 y h=0.001. Se observa que cuando h se reduce
los polinomios P7 y P8 se rompen por completo. Esto sucede debido a que en cada iteración para calcular 
la derivada central estámos dividiendo por un factor porporcional a h^8 que aproximadamente seria 10^{-24}
y sabemos que la memoria estándar de la computadora (el punto flotante de 64 bits) 
solo tiene una precisión que llega a detectar diferencias hasta aproximadamente 10^{-16}

"""
