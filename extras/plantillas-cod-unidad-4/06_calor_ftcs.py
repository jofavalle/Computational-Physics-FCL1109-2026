"""
PLANTILLA 06 - Ecuación de calor (difusión térmica) - Esquema FTCS
====================================================================
Ecuación diferencial:
    ∂T/∂t = α ∂²T/∂x²

donde α [m²/s] es la difusividad térmica.

Discretización FTCS (Forward Time, Centered Space):
    T_nuevo[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
    donde r = α*dt/dx²

⚠ CONDICIÓN DE ESTABILIDAD (Von Neumann):
    r = α*dt/dx² ≤ 0.5
  Si r > 0.5 la solución oscila y diverge.

Solución estacionaria (condiciones Dirichlet T=T_izq en x=0, T=T_der en x=L):
    T_est(x) = T_izq + (T_der - T_izq) * x / L

Fuente: referencias/cap04_ecuaciones_onda_fluidos.md
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L     = 1.0    # [m]     longitud de la barra
alpha = 0.01   # [m²/s]  difusividad térmica
T_izq = 100.0  # [°C]    temperatura en x=0 (Dirichlet)
T_der =   0.0  # [°C]    temperatura en x=L (Dirichlet)
T_ini =   0.0  # [°C]    temperatura inicial uniforme

# =============================================================================
# GRILLA ESPACIAL Y TEMPORAL
# =============================================================================
Nx = 100
dx = L / (Nx - 1)
# Elegir dt para que r = 0.4 (seguro)
r_target = 0.4
dt       = r_target * dx**2 / alpha
Nt       = 3000

# Verificación
r = alpha * dt / dx**2
assert r <= 0.5, f"¡Inestable! r = {r:.4f} > 0.5. Reduce dt."
print(f"r = {r:.4f}  (Von Neumann estable ✓)")
print(f"dt = {dt:.6f} s,  tiempo total = {Nt*dt:.4f} s")

x = np.linspace(0, L, Nx)

# =============================================================================
# CONDICIÓN INICIAL
# =============================================================================
T = np.full(Nx, T_ini)   # temperatura inicial uniforme

# Aplicar condiciones de frontera (se mantienen fijas todo el tiempo)
T[0]  = T_izq
T[-1] = T_der

# =============================================================================
# ALGORITMO CENTRAL - FTCS
# =============================================================================
pasos_guardar = {0, Nt//10, Nt//4, Nt//2, Nt - 1}
instantaneas  = {0: T.copy()}

for n in range(1, Nt):
    T_nuevo = T.copy()

    # nodos interiores: diferencias finitas centradas en espacio
    T_nuevo[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])

    # condiciones de frontera Dirichlet (fijas)
    T_nuevo[0]  = T_izq
    T_nuevo[-1] = T_der

    T = T_nuevo

    if n in pasos_guardar:
        instantaneas[n] = T.copy()

# =============================================================================
# SOLUCIÓN ESTACIONARIA ANALÍTICA
# =============================================================================
T_estacionaria = T_izq + (T_der - T_izq) * x / L

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: evolución temporal
ax = axes[0]
for paso, T_snap in instantaneas.items():
    t_snap = paso * dt
    ax.plot(x, T_snap, label=f't = {t_snap:.3f} s')
ax.plot(x, T_estacionaria, 'k--', linewidth=2, label='Estacionaria (analítica)')
ax.set_xlabel('x [m]')
ax.set_ylabel('T [°C]')
ax.set_title(f'Difusión FTCS  (α={alpha}, r={r:.2f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel derecho: error respecto al estado estacionario
ax2 = axes[1]
for paso, T_snap in instantaneas.items():
    if paso == 0:
        continue
    t_snap = paso * dt
    error  = np.abs(T_snap - T_estacionaria)
    ax2.semilogy(x, error + 1e-14, label=f't = {t_snap:.3f} s')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('|T - T_est| [°C]')
ax2.set_title('Error respecto a solución estacionaria')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
