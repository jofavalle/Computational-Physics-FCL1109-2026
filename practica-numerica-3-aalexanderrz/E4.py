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

# Ejercicio 4: Conservación 
v0_ej4 = 0.1 # Velocidad pequeña para evitar error de división por cero
U_ej4 = fisicas_a_conservativas(rho0 + A*rho0*w, v0_ej4 + A*cs0*w, p0 + A*gamma*p0*w)

M0 = np.sum(U_ej4[0]) * dx
P0 = np.sum(U_ej4[1]) * dx
E0 = np.sum(U_ej4[2]) * dx

t, t_final_ej4 = 0.0, 0.5
tiempos, errores_M, errores_P, errores_E = [t], [0.0], [0.0], [0.0]

while t < t_final_ej4:
    rho_act, v_act, p_act = conservativas_a_fisicas(U_ej4)
    dt = C_cfl * dx / np.max(np.abs(v_act) + np.sqrt(gamma * p_act / rho_act))
    
    F_ej4 = calcular_flujos(U_ej4)
    U_next = np.zeros_like(U_ej4)
    U_next[:, 1:-1] = 0.5*(U_ej4[:, 2:] + U_ej4[:, :-2]) - (dt/(2*dx))*(F_ej4[:, 2:] - F_ej4[:, :-2])
    U_next[:, 0] = 0.5*(U_ej4[:, 1] + U_ej4[:, -2]) - (dt/(2*dx))*(F_ej4[:, 1] - F_ej4[:, -2])
    U_next[:, -1] = U_next[:, 0]
    U_ej4 = U_next.copy()
    t += dt
    
    tiempos.append(t)
    errores_M.append(abs(np.sum(U_ej4[0])*dx - M0) / abs(M0))
    errores_P.append(abs(np.sum(U_ej4[1])*dx - P0) / abs(P0))
    errores_E.append(abs(np.sum(U_ej4[2])*dx - E0) / abs(E0))

plt.figure(figsize=(10, 5))
plt.plot(tiempos, errores_M, label='Error Relativo Masa')
plt.plot(tiempos, errores_P, label='Error Relativo Momento', linestyle='--')
plt.plot(tiempos, errores_E, label='Error Relativo Energía', linestyle='-.')
plt.title('Ejercicio 4: Errores de Conservación en el Tiempo')
plt.xlabel('Tiempo t')
plt.ylabel('Error Relativo')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

print(f"Error Final L2 Masa: {errores_M[-1]:.3e}")