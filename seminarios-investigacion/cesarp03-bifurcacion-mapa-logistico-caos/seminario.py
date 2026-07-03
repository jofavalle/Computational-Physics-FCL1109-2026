import numpy as np
import matplotlib.pyplot as plt

def mapa_logistico_serie(mu, x0, generaciones):
    """Calcula la serie de tiempo para el mapa logístico."""
    x = np.zeros(generaciones)
    x[0] = x0
    for i in range(1, generaciones):
        x[i] = mu * x[i-1] * (1 - x[i-1])
    return x

# Parámetros de la simulación
generaciones = 30
x0_1 = 0.5 # Semilla inicial 1
x0_2 = 0.1 # Semilla inicial 2 (para probar el inciso 'e' de transitorios)

# Valores de mu seleccionados para mostrar los distintos comportamientos (6 valores en total)
mu_values = [0.5, 1.5, 2.8, 3.3, 3.8, 3.9]

# Definimos el tamaño de la figura general
plt.figure(figsize=(14, 8))

for idx, mu in enumerate(mu_values):
    # '2, 3' significa: 2 filas y 3 columnas (capacidad total para 6 subgráficos)
    plt.subplot(2, 3, idx + 1)
    
    # Calculamos con dos semillas diferentes para comparar los transitorios
    serie_1 = mapa_logistico_serie(mu, x0_1, generaciones)
    serie_2 = mapa_logistico_serie(mu, x0_2, generaciones)
    
    # Graficamos ambas series de tiempo
    plt.plot(range(generaciones), serie_1, 'b-', label=f'$x_0$={x0_1}', alpha=0.7)
    plt.plot(range(generaciones), serie_2, 'r--', label=f'$x_0$={x0_2}', alpha=0.7)
    
    # Formato de cada subgráfico
    plt.title(rf'$\mu = {mu}$')
    plt.xlabel('Generación (i)')
    plt.ylabel('$x_i$')
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.6) # Añadimos una cuadrícula tenue para mejor visualización
    
    # Colocamos la leyenda solo en el primer recuadro para no saturar la imagen
    if idx == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# Verificación numérica para mu = 3.2 (Debería oscilar en un ciclo de periodo 2)
print("--- Verificación Numérica para mu = 3.2 ---")
mu_2ciclos = 3.2
x = 0.1 # Semilla arbitraria

# Iteramos 100 veces para "matar" los transitorios iniciales
for i in range(100):
    x = mu_2ciclos * x * (1 - x)

# Mostramos el estado asintótico en las generaciones siguientes
print(f"Generación 101: {x:.4f}")
x_sig = mu_2ciclos * x * (1 - x)
print(f"Generación 102: {x_sig:.4f}")
x_sig2 = mu_2ciclos * x_sig * (1 - x_sig)
print(f"Generación 103: {x_sig2:.4f} (Se repite el valor de la gen 101)")