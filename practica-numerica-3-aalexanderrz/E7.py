import numpy as np
import matplotlib.pyplot as plt

# Parámetros y Funciones Base 
gamma, rho0, p0 = 5/3, 1.0, 1.0
cs0 = np.sqrt(gamma * p0 / rho0)

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

# Ejercicio 7: Régimen No Lineal 
A_valores = [1e-3, 1e-2, 1e-1]
t_final_ej7 = 0.5

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

for idx, A_test in enumerate(A_valores):
    U_ej7 = fisicas_a_conservativas(rho0 + A_test*rho0*w, A_test*cs0*w, p0 + A_test*gamma*p0*w)
    t = 0.0
    
    while t < t_final_ej7:
        rho_act, v_act, p_act = conservativas_a_fisicas(U_ej7)
        dt = C_cfl * dx / np.max(np.abs(v_act) + np.sqrt(np.abs(gamma * p_act / rho_act)))
        if t + dt > t_final_ej7: dt = t_final_ej7 - t
            
        F_ej7 = calcular_flujos(U_ej7)
        U_next = np.zeros_like(U_ej7)
        U_next[:, 1:-1] = 0.5*(U_ej7[:, 2:] + U_ej7[:, :-2]) - (dt/(2*dx))*(F_ej7[:, 2:] - F_ej7[:, :-2])
        U_next[:, 0] = 0.5*(U_ej7[:, 1] + U_ej7[:, -2]) - (dt/(2*dx))*(F_ej7[:, 1] - F_ej7[:, -2])
        U_next[:, -1] = U_next[:, 0]
        U_ej7 = U_next.copy()
        t += dt
        
    rho_final = U_ej7[0]
    rho_rel_norm = (rho_final - rho0) / (A_test * rho0)
    
    axs[idx].plot(x, w, 'k--', label='Inicial Normalizada')
    axs[idx].plot(x, rho_rel_norm, 'r-', linewidth=2, label='Final Normalizada')
    axs[idx].set_title(f'Amplitud $A = 10^{{{int(np.log10(A_test))}}}$')
    axs[idx].set_xlabel('Posición x')
    axs[idx].legend()
    axs[idx].grid(True)

plt.suptitle("Ejercicio 7: Evolución hacia el Régimen No Lineal", fontsize=16)
plt.tight_layout()
plt.show()