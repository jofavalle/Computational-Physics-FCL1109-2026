import numpy as np
import matplotlib.pyplot as plt

# Parámetros del diagrama según el libro
mu_min, mu_max = 1.0, 4.0
pasos_mu = 1000
transitorios = 200  # Iteraciones para estabilizar (transients)
iteraciones_guardadas = 200  # Puntos a guardar tras estabilizarse

# Arreglos para guardar los datos a graficar
mu_lista = []
x_lista = []

# Arreglo de valores mu (los 1000 pasos o "bins")
mu_valores = np.linspace(mu_min, mu_max, pasos_mu)

for mu in mu_valores:
    # Semilla inicial (inciso 8.c pide variar, pero por simplicidad una semilla fija 
    # suele ser suficiente si no hay atractores múltiples coexistentes, 
    # aunque aquí usamos un arreglo aleatorio pequeño para robustez)
    x = np.random.random() 
    
    # 1. Bucle para matar los transitorios (no guardamos datos)
    for _ in range(transitorios):
        x = mu * x * (1 - x)
        
    # 2. Bucle para guardar los valores estables (atractores)
    for _ in range(iteraciones_guardadas):
        x = mu * x * (1 - x)
        # Redondeamos a 4 decimales como sugiere el libro (8.d) para evitar ruido visual
        x_truncado = round(x, 4)
        mu_lista.append(mu)
        x_lista.append(x_truncado)

# 3. Graficar (8.e y 8.f)
plt.figure(figsize=(10, 6), dpi=150)
# Usamos puntos muy pequeños (s=0.1), color negro y sin conectarlos
plt.scatter(mu_lista, x_lista, s=0.1, color='black', marker='.', alpha=0.2)

plt.title('Diagrama de Bifurcación del Mapa Logístico', fontsize=16)
plt.xlabel(r'Tasa de crecimiento ($\mu$)', fontsize=14)
plt.ylabel(r'Población estable ($x_*$)', fontsize=14)
plt.xlim(1.0, 4.0)
plt.ylim(0, 1)

# Para mostrar el diagrama principal
plt.tight_layout()
plt.show()

# Nota para la presentación: Puedes cambiar mu_min a 3.5 y mu_max a 4.0 
# en el código para hacer el "zoom" que pide el inciso 8.f (autosimilitud).