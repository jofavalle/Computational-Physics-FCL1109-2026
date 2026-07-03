# Código del seminario para Física Computacional Junio - 2026
# Simulador de estados coherentes de Glauber para el oscilador armónico hasta un número n
# Basado en el código de R. Landau

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.special as sp

# Parámetros constantes
sqpi = np.sqrt(np.pi)
E = 3
alpha = np.sqrt(E - 0.5)
factr = np.exp(- 0.5 * alpha ** 2)
nmax1 = 20 # cantidad de estados considerados
nmax2 = 5

# Funciones necesarias
def Hermite(x, n):
    if (n == 0):
        p = 1.0
    elif (n == 1):
        p = 2 * x 
    else:
        p0 = 1
        p1 = 2 * x
        for i in range(1, n):
            p2 = 2 * x * p1 - 2 * i * p0
            p0 = p1
            p1 = p2
            p =p2
    return p

def glauber(x, t, nmax):
    Reterm = 0.0
    Imterm = 0.0
    factr = np.exp(- 0.5 * alpha ** 2)
    for n in range(0, nmax):
        fact = np.sqrt(1.0 / (sp.gamma(n + 1) * sqpi * (2 ** n)))
        psin = fact * Hermite(x, n) * np.exp(- 0.5 * x ** 2)
        den = np.sqrt(sp.gamma(n + 1))
        num = factr * (alpha ** n) * psin
        Reterm += num * (np.cos((n + 0.5) * t)) / den
        Imterm += num * (np.sin((n + 0.5) * t)) / den
    phi = np.sqrt(Reterm ** 2 + Imterm ** 2)
    return phi

# Elementos necesariso para la animación
xx = np.arange(- 6.0, 6.0, 0.2)
time = np.arange(0, 6, 0.05)
fig, ax = plt.subplots(1, 2, figsize = (12, 5))
values1 = np.zeros((len(xx), len(time)))
values2 = np.zeros_like(values1)
for i in range(len(time)):
    for j in range(len(xx)):
        values1[j, i] = glauber(xx[j], time[i], nmax1)
        values2[j, i] = glauber(xx[j], time[i], nmax2)
y1, = ax[0].plot(xx, glauber(xx, time[0], nmax1))
y2, = ax[1].plot(xx, glauber(xx, time[0], nmax2))

# Función para realizar animación
def animate(frame):
    y1.set_ydata(values1[:, frame])
    y2.set_ydata(values2[:, frame])
    return y1, y2,

ax[0].grid()
ax[0].set_title('Gauber states at different times for n = 5')
ax[0].set_xlabel('x')
ax[0].set_ylabel('$|\psi(x, t)|^2$')
ax[1].grid()
ax[1].set_title('Gauber states at different times for n = 20')
ax[1].set_xlabel('x')
ax[1].set_ylabel('$|\psi(x, t)|^2$')
animation = FuncAnimation(fig, animate, int(len(time)), interval = 33, blit = True)
#animation.save('glauber_state.animation.gif', writer='pillow', fps=5)

plt.show()
