import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parámetros y Funciones Base ---
gamma, rho0, p0 = 5/3, 1.0, 1.0
cs0 = np.sqrt(gamma * p0 / rho0)
A = 2.1e-4
x0, xf = 0.0, 1.0
C_cfl = 0.5

def fisicas_a_conservativas(rho, v, p):
    return np.array([rho, rho * v, p / (gamma - 1) + 0.5 * rho * v**2])

def conservativas_a_fisicas(U):
    rho, v = U[0], U[1] / U[0]
    return rho, v, (gamma - 1) * (U[2] - 0.5 * rho * v**2)

def calcular_flujos(U):
    rho, v, p = conservativas_a_fisicas(U)
    return np.array([rho * v, rho * v**2 + p, (U[2] + p) * v])

# Ejercicio 5: Convergencia 
N_valores = [100, 200, 400, 800]
t_final_ej5 = 0.5
errores_L2, dx_valores = [], []

print("Calculando simulaciones para Análisis de Convergencia...")

for N_actual in N_valores:
    x_n = np.linspace(x0, xf, N_actual)
    dx_n = (xf - x0) / (N_actual - 1)
    dx_valores.append(dx_n)
    
    w_n = np.cos(2 * np.pi * (x_n - x0) / (xf - x0) + 5 * np.pi / 8)
    U_ej5 = fisicas_a_conservativas(rho0 + A*rho0*w_n, 0.0 + A*cs0*w_n, p0 + A*gamma*p0*w_n)
    t = 0.0
    
    while t < t_final_ej5:
        rho_act, v_act, p_act = conservativas_a_fisicas(U_ej5)
        dt = C_cfl * dx_n / np.max(np.abs(v_act) + np.sqrt(gamma * p_act / rho_act))
        if t + dt > t_final_ej5: dt = t_final_ej5 - t
            
        F_ej5 = calcular_flujos(U_ej5)
        U_next = np.zeros_like(U_ej5)
        U_next[:, 1:-1] = 0.5*(U_ej5[:, 2:] + U_ej5[:, :-2]) - (dt/(2*dx_n))*(F_ej5[:, 2:] - F_ej5[:, :-2])
        U_next[:, 0] = 0.5*(U_ej5[:, 1] + U_ej5[:, -2]) - (dt/(2*dx_n))*(F_ej5[:, 1] - F_ej5[:, -2])
        U_next[:, -1] = U_next[:, 0]
        U_ej5 = U_next.copy()
        t += dt
        
    rho_exacta = rho0 + A * rho0 * np.cos(2 * np.pi * (x_n - cs0 * t_final_ej5 - x0) / (xf - x0) + 5 * np.pi / 8)
    rho_final, _, _ = conservativas_a_fisicas(U_ej5)
    errores_L2.append(np.sqrt(np.sum((rho_final - rho_exacta)**2) / N_actual))

orden = np.polyfit(np.log(dx_valores), np.log(errores_L2), 1)[0]

plt.figure(figsize=(9, 6))
plt.plot(N_valores, errores_L2, 'bo-', label='Error L2 Numérico')
plt.plot(N_valores, np.array(dx_valores) * (errores_L2[0]/dx_valores[0]), 'k--', label='Tendencia teórica O(dx)')
plt.xscale('log'); plt.yscale('log')
plt.title(f'Ejercicio 5: Convergencia de Lax-Friedrichs')
plt.xlabel('Número de Nodos (N)')
plt.ylabel('Error L2')
plt.xticks(N_valores, labels=[str(n) for n in N_valores])
plt.legend(); plt.grid(True)
plt.show()

print(f"Orden de convergencia empírico calculado: {orden:.3f}")