import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import mpmath as mp

# Listing 6.17 (Landau) — Funciones de onda regulares de dispersión de Coulomb
# Caso: partícula alfa sobre núcleo de oro (Elab = 7 MeV)
# Unidades: masas en MeV/c^2, hbar*c en MeV·fm, r en fm

# Arreglos para almacenar soluciones parciales (l = 0..9)
f_l = np.zeros(10, dtype=complex)
parte_real = np.zeros((10, 161), dtype=float)

# Unidad imaginaria y parámetros físicos del sistema
zi = complex(0.0, 1.0)
masa_oro = 196.966569 * 931.494
masa_alfa = 4.002602 * 931.494
Z_oro = 79
Z_alfa = 2

# Masa reducida del sistema alfa-oro
masa_reducida = masa_alfa * masa_oro / (masa_alfa + masa_oro)

# Constantes y energía de laboratorio
hbar_c = 197.33
energia_lab = 7.0

# Conversión a energía en centro de masa
energia_cm = energia_lab * masa_oro / (masa_alfa + masa_oro)

# Velocidad relativa y número de onda
velocidad = np.sqrt(2.0 * energia_cm / masa_reducida)
kappa = np.sqrt(2.0 * masa_reducida * energia_cm) / hbar_c

# Parámetro de Coulomb (eta) y factor exponencial de normalización
eta_coulomb = Z_alfa * Z_oro * masa_reducida / (hbar_c * kappa * 137.0)
expo_factor = np.exp(-0.5 * eta_coulomb * np.pi)

# Bucle radial: réplica fiel del esquema de Landau
indice_r = 0
for radio in np.arange(0.1, 80.5, 0.5):
    rho = complex(0.0, -2.0 * kappa * radio)          # -2ikr
    exp_ikr = complex(np.cos(kappa * radio), np.sin(kappa * radio))

    # Bucle en momento angular orbital l = 0..9
    for l in range(0, 10):
        a_param = l + 1.0 + eta_coulomb * zi
        sol_hipergeom = mp.hyp1f1(a_param, 2 * l + 2.0, rho)
        rho_pot_l = (-rho) ** l
        gamma_a = special.gamma(a_param)

        # Amplitud radial compleja regular de Coulomb
        u_parcial = rho_pot_l * exp_ikr * complex(sol_hipergeom) * gamma_a * expo_factor / float(mp.factorial(2 * l))
        f_l[l] = u_parcial / np.sqrt(velocidad)
        parte_real[l, indice_r] = np.real(f_l[l])

    indice_r += 1

# Graficar las primeras tres ondas parciales: S, P y D
r_vals = np.arange(0.1, 80.5, 0.5)
plt.figure(figsize=(9, 5))
plt.plot(r_vals, parte_real[0, :], label='S (l=0)')
plt.plot(r_vals, parte_real[1, :], label='P (l=1)', linewidth=2)
plt.plot(r_vals, parte_real[2, :], label='D (l=2)', linewidth=3)
plt.xlabel('r [fm]')
plt.ylabel('Re[y_l(r)]')
plt.title('Funciones radiales de Coulomb (dispersión alfa-oro)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()