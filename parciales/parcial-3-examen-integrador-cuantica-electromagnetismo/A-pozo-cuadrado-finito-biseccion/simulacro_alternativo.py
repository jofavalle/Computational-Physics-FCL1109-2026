import numpy as np
import matplotlib.pyplot as plt

# Parámetros del pozo
V_0 = 25.0 # profundidad del pozo
a = 1.0 # ancho del pozo

# Definimos las funciones trascendentes

def f_par(E_B):
    u = np.sqrt(V_0 - E_B)
    return u * np.tan(u) - np.sqrt(E_B)

def f_impar(E_B):
    u = np.sqrt(V_0 - E_B)
    return -u / np.tan(u) - np.sqrt(E_B)

# Definimos la función biseccion

def biseccion(f,a,b,eps=1e-12, Nmax=200):
    fa = f(a)
    for i in range(Nmax):
        c = 0.5*(a+b)
        fc = f(c)
        if abs(fc) < eps or 0.5*(b-a) < eps:
            return c
        if fa*fc < 0.0:
            b = c
        else:
            a = c
            fa = fc
    return 0.5*(a+b)

# Buscamos las raíces

N_scan = 40000 # Número de puntos del barrido
E_scan = np.linspace(1e-7, V_0 - 1e-7, N_scan) # Barrido de energía, array de N_scan puntos entre 0 y V_0
umbral_salto = 50.0 # Umbral para detectar saltos en las funciones trascendentes

raices_par, raices_impar = [], [] # Listas para almacenar las raíces encontradas

for i in range(N_scan - 1): # N_scan-1 porque estamos trabajando con los subintervalos
    a,b = E_scan[i], E_scan[i+1] # Extremos del subintervalo i-ésimo
    fpa, fpb = f_par(a), f_par(b)
    if fpa*fpb < 0.0 and abs(fpa - fpb) < umbral_salto:
        raices_par.append(biseccion(f_par,a,b))
    fia, fib = f_impar(a), f_impar(b)
    if fia*fib < 0.0 and abs(fia - fib) < umbral_salto:
        raices_impar.append(biseccion(f_impar,a,b))

# Construimos una lista de pares energía, etiqueta para ordenar las raíces por energía con signo cambiado
# de mayor a menor
todos = sorted([(eb, 'par') for eb in raices_par] + [(eb, 'impar') for eb in raices_impar], key=lambda x: -x[0])

print("=" * 60)
print("Raíces encontradas (de mayor a menor):")
print("=" * 60)
print(f'{"n "} {"Paridad "} {"E_B "} {"E = -E_B"}')
for n, (eb, paridad) in enumerate(todos): # Enumerate entrega pares (n, (eb, paridad))
    print(f'{n} {" "} {paridad} {" "} {eb:.8f} {" "} {-eb:.8f}')
print(f'\nNúmero total de estados ligados: {len(todos)}')
print("Esperado 4")