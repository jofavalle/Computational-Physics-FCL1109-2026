import numpy as np
import matplotlib.pyplot as plt

#  Parámetros y Funciones Base 
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

# Ejercicio 1: Ondas Sonoras
v0 = 0.0
rho_init = rho0 + A * rho0 * w
p_init = p0 + A * gamma * p0 * w
v_init = v0 + A * cs0 * w

U = fisicas_a_conservativas(rho_init, v_init, p_init)
t, t_final = 0.0, 0.5

while t < t_final:
    rho_act, v_act, p_act = conservativas_a_fisicas(U)
    dt = C_cfl * dx / np.max(np.abs(v_act) + np.sqrt(gamma * p_act / rho_act))
    if t + dt > t_final: dt = t_final - t
    
    F = calcular_flujos(U)
    U_next = np.zeros_like(U)
    U_next[:, 1:-1] = 0.5 * (U[:, 2:] + U[:, :-2]) - (dt / (2 * dx)) * (F[:, 2:] - F[:, :-2])
    U_next[:, 0] = 0.5 * (U[:, 1] + U[:, -2]) - (dt / (2 * dx)) * (F[:, 1] - F[:, -2])
    U_next[:, -1] = U_next[:, 0]
    U = U_next.copy()
    t += dt

rho_fin, _, _ = conservativas_a_fisicas(U)
rho_relativa_num = (rho_fin - rho0) / rho0
rho_relativa_ana = A * np.cos(2 * np.pi * (x - cs0 * t_final - x0) / (xf - x0) + 5 * np.pi / 8)

plt.figure(figsize=(10, 5))
plt.plot(x, rho_relativa_num, 'b-', label='Solución Numérica', linewidth=2)
plt.plot(x, rho_relativa_ana, 'r--', label='Solución Analítica', linewidth=2)
plt.title(f'Ejercicio 1: Analítica vs Numérica a t = {t_final:.2f}')
plt.xlabel('Posición x')
plt.ylabel('Amplitud Relativa')
plt.legend()
plt.grid(True)
plt.show()

distancia = x[np.argmax(rho_fin)] - x[np.argmax(rho_init)]
if distancia < 0: distancia += (xf - x0)
print(f"Velocidad de propagación numérica: {distancia / t_final:.4f} (Teórica cs0 = {cs0:.4f})")