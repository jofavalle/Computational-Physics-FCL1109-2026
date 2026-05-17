import fcl1109 as fcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fuerza corregida: F = (-k/r^2 + C/r^3) r_hat
def derivatives(k, C):
    def f(t, state):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        ax = (-k / r**3 + C / r**4) * x
        ay = (-k / r**3 + C / r**4) * y
        return np.array([vx, vy, ax, ay])
    return f

def simular_C(estado_inicial, k, C):
    f = derivatives(k, C)
    states = np.zeros((n_steps, 4))
    states[0] = estado_inicial
    for i in range(1, n_steps):
        states[i] = fcl.rk4(f, t_values[i-1], states[i-1], dt)
    return states

# Condiciones iniciales y parámetros
k = 1.0
C = [0.008, 0.03, 0]  # Diferentes valores de C para comparar
labels = ['C=0.008', 'C=0.03', 'C=0']
initial_state = np.array([0.5, 0.0, 0.0, 1.5])  # Posición inicial (x, y) y velocidad inicial (vx, vy)
t_inicial = 0.0
t_final = 100.0
dt = 0.001
t_values = np.arange(t_inicial, t_final, dt)
n_steps = len(t_values)

# Simulación para cada valor de C
trajectories = []
for C_value in C:
    trajectory = simular_C(initial_state, k, C_value)
    trajectories.append(trajectory)

# Graficar las trayectorias
plt.figure(figsize=(10, 8))
for i, trajectory in enumerate(trajectories):
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=labels[i])
plt.title('Trayectorias para diferentes valores de C')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()