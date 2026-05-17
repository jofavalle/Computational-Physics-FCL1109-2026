import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 250
dx = 0.4
dt = 0.05
eps = 0.2
mu = 0.1
Nt = 200

x = np.arange(N)*dx
t_vals = np.arange(Nt)*dt

# Condición inicial de ruido
np.random.seed(0)  # Para reproducibilidad
u = 0.2*np.random.rand(N)

# Suavizado leve
u = (np.roll(u, 1) + u + np.roll(u, -1)) / 3
u_old = np.copy(u)
u_new = np.zeros_like(u)

# Almacenamiento 3D
U = np.zeros((Nt, N))

# Evolución KdV
for n in range(Nt):
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i+1] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])
        u_new[i] = u_old[i] - eps*dt/(3*dx)*nonlinear - mu*dt/(dx**3)*dispersion

    # Condiciones de frontera
    u_new[0:2] = 0.0
    u_new[-2:] = 0.0
    
    # Actualizar
    u_old = np.copy(u)
    u = np.copy(u_new)

    U[n, :] = u

# Malla 3D
X, T = np.meshgrid(x, t_vals)

#Gráfica 3D
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, T, U, cmap='viridis', linewidth=0, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x, t)')
ax.set_title('Evolución de u(x, t) con condiciones iniciales de ruido (KdV)')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Amplitud')

plt.show()