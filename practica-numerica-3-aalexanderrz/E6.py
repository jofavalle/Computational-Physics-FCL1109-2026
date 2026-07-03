import numpy as np
import matplotlib.pyplot as plt
import warnings

#  Parámetros y Funciones Base 
gamma, rho0, p0 = 5/3, 1.0, 1.0
cs0 = np.sqrt(gamma * p0 / rho0)
A = 2.1e-4

x0, xf, N = 0.0, 1.0, 100
x = np.linspace(x0, xf, N)
dx = (xf - x0) / (N - 1)
w = np.cos(2 * np.pi * (x - x0) / (xf - x0) + 5 * np.pi / 8)

def fisicas_a_conservativas(rho, v, p):
    return np.array([rho, rho * v, p / (gamma - 1) + 0.5 * rho * v**2])

def conservativas_a_fisicas(U):
    rho, v = U[0], U[1] / U[0]
    return rho, v, (gamma - 1) * (U[2] - 0.5 * rho * v**2)

def calcular_flujos(U):
    rho, v, p = conservativas_a_fisicas(U)
    return np.array([rho * v, rho * v**2 + p, (U[2] + p) * v])

# Ejercicio 6: Estabilidad Numérica (CFL)
cfl_test_values = [0.5, 0.9, 1.0, 1.05]
t_final_ej6 = 0.15

plt.figure(figsize=(12, 7))
print("Probando estabilidad para distintos valores CFL...")

for cfl_actual in cfl_test_values:
    U_ej6 = fisicas_a_conservativas(rho0 + A*rho0*w, A*cs0*w, p0 + A*gamma*p0*w)
    t, estable = 0.0, True
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while t < t_final_ej6:
            rho_act, v_act, p_act = conservativas_a_fisicas(U_ej6)
            dt = cfl_actual * dx / np.max(np.abs(v_act) + np.sqrt(np.abs(gamma * p_act / rho_act)))
            
            F_ej6 = calcular_flujos(U_ej6)
            U_next = np.zeros_like(U_ej6)
            U_next[:, 1:-1] = 0.5*(U_ej6[:, 2:] + U_ej6[:, :-2]) - (dt/(2*dx))*(F_ej6[:, 2:] - F_ej6[:, :-2])
            U_next[:, 0] = 0.5*(U_ej6[:, 1] + U_ej6[:, -2]) - (dt/(2*dx))*(F_ej6[:, 1] - F_ej6[:, -2])
            U_next[:, -1] = U_next[:, 0]
            U_ej6 = U_next.copy()
            t += dt
            
            if np.any(np.isnan(U_ej6)) or np.max(np.abs(U_ej6[0])) > 10 * rho0:
                estable = False; break
                
    if estable:
        print(f"CFL = {cfl_actual:.2f} -> ESTABLE")
        if cfl_actual in [0.5, 1.0]: 
            plt.plot(x, (U_ej6[0] - rho0)/rho0, label=f'Estable (CFL={cfl_actual})')
    else:
        print(f"CFL = {cfl_actual:.2f} -> INESTABLE (Colapso numérico)")
        plt.plot(x, (U_ej6[0] - rho0)/rho0, '--', label=f'Inestable (CFL={cfl_actual})')

plt.title('Ejercicio 6: Efecto del Número CFL en la Estabilidad Numérica')
plt.xlabel('Posición x')
plt.ylabel('Perturbación Relativa')
plt.ylim([-3e-4, 3e-4]) 
plt.legend(); plt.grid(True)
plt.show()