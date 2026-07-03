import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros físicos
c = 1.0  # Velocidad de la luz
Nz = 300  # Número de puntos en la dirección z
dz = 1.0  # Paso espacial
beta = 0.4 # Condición de Courant para la estabilidad numérica
dt = beta * dz / c  # Paso de tiempo

steps = 600  # Número de pasos de tiempo

# Campos
E_x = np.zeros(Nz)  # Campo eléctrico
B_y = np.zeros(Nz)  # Campo magnético

# Pulso inicial gaussiano
z = np.arange(Nz)
E_x = np.exp(-((z - 80) / 15)**2)

# Figura
fig, ax = plt.subplots(figsize=(10, 5))
lineE, = ax.plot(z, E_x, label='E_x')
lineB, = ax.plot(z, B_y, label='B_y')
ax.set_xlim(0, Nz)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('z')
ax.set_ylabel('Campo')
ax.legend()

# Actualización FDTD
def update(frame):
    global E_x, B_y
    
    # Actualizar campo eléctrico
    E_x[1:-1] += beta*(B_y[:-2] - B_y[1:-1])
    
    # Actualizar campo magnético
    B_y[1:-1] += -beta*(E_x[2:] - E_x[1:-1])

    #Condiciones de frontera
    E_x[0] = E_x[-1] = 0
    B_y[0] = B_y[-1] = 0
    
    lineE.set_ydata(E_x)
    lineB.set_ydata(B_y)
    return lineE, lineB

ani = FuncAnimation(
    fig, update, frames=steps, interval=20, blit=True
)

plt.show()