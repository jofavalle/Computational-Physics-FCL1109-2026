import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros y Funciones Base 
gamma, rho0, p0 = 5/3, 1.0, 1.0
cs0 = np.sqrt(gamma * p0 / rho0)
A = 2.1e-4

x0, xf, N = 0.0, 1.0, 100
x = np.linspace(x0, xf, N)
dx = (xf - x0) / (N - 1)
C_cfl = 0.5
w = np.cos(2 * np.pi * (x - x0) / (xf - x0) + 5 * np.pi / 8)

def fisicas_a_conservativas(rho, v, p):
    return np.array([rho, rho * v, p / (gamma - 1) + 0.5 * rho * v**2])

def conservativas_a_fisicas(U):
    rho, v = U[0], U[1] / U[0]
    return rho, v, (gamma - 1) * (U[2] - 0.5 * rho * v**2)

def calcular_flujos(U):
    rho, v, p = conservativas_a_fisicas(U)
    return np.array([rho * v, rho * v**2 + p, (U[2] + p) * v])

# Ejercicio 3: Propagación en Sentido Opuesto 
plt.figure(figsize=(10, 5))
plt.plot(x, A * w, 'k--', label='Posición Inicial')


U_ej3 = fisicas_a_conservativas(rho0 + A*rho0*w, 0.0 - A*cs0*w, p0 + A*gamma*p0*w)
t, t_final_ej3 = 0.0, 0.5

while t < t_final_ej3:
    rho_act, v_act, p_act = conservativas_a_fisicas(U_ej3)
    dt = C_cfl * dx / np.max(np.abs(v_act) + np.sqrt(gamma * p_act / rho_act))
    if t + dt > t_final_ej3: dt = t_final_ej3 - t
    
    F_ej3 = calcular_flujos(U_ej3)
    U_next = np.zeros_like(U_ej3)
    U_next[:, 1:-1] = 0.5*(U_ej3[:, 2:] + U_ej3[:, :-2]) - (dt/(2*dx))*(F_ej3[:, 2:] - F_ej3[:, :-2])
    U_next[:, 0] = 0.5*(U_ej3[:, 1] + U_ej3[:, -2]) - (dt/(2*dx))*(F_ej3[:, 1] - F_ej3[:, -2])
    U_next[:, -1] = U_next[:, 0]
    U_ej3 = U_next.copy()
    t += dt

rho_final, _, _ = conservativas_a_fisicas(U_ej3)
plt.plot(x, (rho_final - rho0) / rho0, 'g-', linewidth=2, label='Onda desplazada a la izquierda')

plt.title('Ejercicio 3: Propagación de Onda en Sentido Opuesto')
plt.xlabel('Posición x')
plt.ylabel('Perturbación Relativa')
plt.legend()
plt.grid(True)
plt.show()