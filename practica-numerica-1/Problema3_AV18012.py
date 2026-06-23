# Practica Numerica 1 - Problema 3
# Fisica Computacional, Ciclo I 2026
# Carnet: AV18012
# Ecuaciones de Lorenz con Runge-Kutta de orden 2 (punto medio)

import numpy as np
import matplotlib.pyplot as plt

sigma = 10.0
r = 28.0
b = 8.0/3.0


def lorenz(t, s):
    x, y, z = s
    return np.array([sigma*(y-x), r*x - y - x*z, x*y - b*z])


# RK2 punto medio
def rk2(f, s0, t0, tf, dt):
    n = int(round((tf-t0)/dt))
    t = np.linspace(t0, t0+n*dt, n+1)
    s = np.empty((n+1, len(s0)))
    s[0] = s0
    for i in range(n):
        k1 = f(t[i], s[i])
        k2 = f(t[i] + dt/2, s[i] + dt/2*k1)
        s[i+1] = s[i] + dt*k2
    return t, s


s0 = np.array([1.0, 0.0, 0.0])   # condiciones iniciales (x,y,z)
t, s = rk2(lorenz, s0, 0.0, 50.0, 0.01)
x, y, z = s[:, 0], s[:, 1], s[:, 2]

# (a) y vs t
plt.figure(figsize=(11, 4.5))
plt.plot(t, y, lw=0.7)
plt.xlabel('t'); plt.ylabel('y')
plt.title('Lorenz: y en funcion del tiempo (RK2, dt = 0.01)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("P3_y_vs_t.png", dpi=130)
plt.close()

# (b) z vs x, aqui deberia salir el atractor extrano
plt.figure(figsize=(7.5, 7))
plt.plot(x, z, lw=0.4, color='tab:red')
plt.xlabel('x'); plt.ylabel('z')
plt.title('Atractor de Lorenz (z vs x)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("P3_atractor.png", dpi=130)
plt.close()
