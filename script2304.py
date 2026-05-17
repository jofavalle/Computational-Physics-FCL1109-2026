import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 71
dx = 1.0/N
dt = 0.001
c = 1.0

r = (c*dt/dx)**2

u = np.zeros((N, N))
u_prev = np.zeros((N, N))
u_next = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        x = i*dx
        y = j*dx
        u[i,j] = np.exp(-100*((x-0.5)**2 + (y-0.5)**2))

u_prev = np.copy(u)

for step in range(200):
    for i in range(1, N-1):
        for j in range(1, N-1):
            u_next[i,j] = 2*u[i,j] - u_prev[i,j] + r*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
    
    u_prev = np.copy(u)
    u = np.copy(u_next)

X, Y = np.meshgrid(range(N), range(N))
plt.imshow(u, cmap='viridis')
plt.colorbar()
plt.title('Membrana vibrante')
plt.show()

# Animación
fig, ax = plt.subplots()
im = ax.imshow(u, animated=True, cmap='viridis')
plt.colorbar(im)

# Función de actualización para la animación
def update(frame):
    global u, u_prev, u_next
    for i in range(1, N-1):
        for j in range(1, N-1):
            u_next[i,j] = 2*u[i,j] - u_prev[i,j] + r*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
    
    # Bordes fijos
    u_next[0,:] = 0
    u_next[-1,:] = 0
    u_next[:,0] = 0
    u_next[:,-1] = 0

    u_prev = np.copy(u)
    u = np.copy(u_next)
    im.set_data(u)
    return [im]

ani = FuncAnimation(fig, update, frames=300, interval=30)
plt.title('Animación de la membrana vibrante')
plt.show()