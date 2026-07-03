import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
dx = 0.4
dt = 0.05
eps = 0.2
mu = 0.1
Nt = 300

x = np.arange(N)*dx

#u = 0.5*(1 - np.tanh(x/5 - 5))
u = 0.2*np.random.rand(N)
u_old = np.copy(u)
u_new = np.zeros_like(u)

posiciones = []
tiempos = []

for n in range(Nt):
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i+1] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])
        u_new[i] = u_old[i] - eps*dt/(3*dx)*nonlinear - mu*dt/(dx**3)*dispersion

    # Actualizar
    u_old = np.copy(u)
    u = np.copy(u_new)

    x_max = x[np.argmax(u)]

    posiciones.append(x_max)
    tiempos.append(n*dt)

# Ajuste lineal
coeficientes = np.polyfit(tiempos, posiciones, 1)
velocidad = coeficientes[0]
print(f"Velocidad de la onda: {velocidad:.4f} unidades por segundo")

plt.figure()
plt.plot(tiempos, posiciones, 'o', label='Datos simulados')
plt.plot(tiempos, np.polyval(coeficientes, tiempos), 'r-', label='Ajuste lineal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición del máximo (unidades)')
plt.title('Posición del máximo de u(x, t) a lo largo del tiempo')
plt.legend()
plt.grid()
plt.show()