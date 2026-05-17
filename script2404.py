import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
dx = 0.4
dt = 0.1
eps = 0.2
mu = 0.1

x = np.arange(N)*dx

u_old = np.zeros(N)
u_new = np.zeros(N)

# u = 0.5*(1 - np.tanh(x/5 - 5))
# u = np.exp(-(x-30)**2/50)
# Doble pulso
# u = np.exp(-(x-20)**2/40) + 0.5*np.exp(-(x-60)**2/40)
# Rectangular
# u = np.zeros_like(x)
# u[60:100] = 1.0
# Ruido aleatorio
u = 0.2*np.random.rand(N)
u_old = np.copy(u)

# Animación 
fig, ax = plt.subplots()
line, = ax.plot(x, u, color='blue')
ax.set_xlim(0, N*dx)
ax.set_ylim(-0.1, 1.1)

def update(frame):
    global u_old, u_new
    for i in range(1, N-1):
        u_new[i] = u_old[i] + eps*(u_old[i+1] - 2*u_old[i] + u_old[i-1]) - mu*u_old[i]*(u_old[i+1] - u_old[i-1])
    u_old, u_new = u_new, u_old
    line.set_ydata(u_old)
    return line,

ani = FuncAnimation(fig, update, frames=200, blit=True)
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Evolución de la función u(x, t)')
plt.grid()
plt.show()