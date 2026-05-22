"""
PLANTILLA 03 — Ecuación de onda con propiedades variables T(x) y ρ(x)
=======================================================================
Ecuación diferencial:
    ρ(x) ∂²y/∂t² = ∂/∂x [T(x) ∂y/∂x] - κ ρ(x) ∂y/∂t

Discretización con T evaluada en semipuntos:
    T_{i+1/2} = 0.5*(T[i] + T[i+1])
    T_{i-1/2} = 0.5*(T[i] + T[i-1])

    y_nuevo[i] = (2*y[i] - y_old[i]
                  + (dt²/ρ[i]) * (T_{i+1/2}*(y[i+1]-y[i]) - T_{i-1/2}*(y[i]-y[i-1])) / dx²
                  - κ*dt*(y[i] - y_old[i]))

⚠ Velocidad variable: v(x) = sqrt(T(x)/ρ(x))
⚠ dt ≤ 0.4 * dx / max(v)   (factor conservador)

MODELOS disponibles:
  - "exponencial": T(x) = T₀ exp(αx),  ρ(x) = ρ₀ exp(αx)
  - "catenaria":   T(x) = T₀ cosh(ρ₀gx/T₀),  ρ(x) = ρ₀ (uniforme)

Fuente: clase_21-04-26.py
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L     = 1.0     # [m]   longitud de la cuerda
alpha = 0.5     # [1/m] coeficiente exponencial (solo modelo exponencial)
rho0  = 0.01    # [kg/m] densidad lineal de referencia
T0    = 40.0    # [N]   tensión de referencia
kappa = 0.0     # [1/s] amortiguamiento

# =============================================================================
# GRILLA Y MODELO DE T(x), ρ(x)
# =============================================================================
Nx = 200
dx = L / (Nx - 1)
x  = np.linspace(0, L, Nx)

modelo = "catenaria"   # "exponencial" o "catenaria"

if modelo == "exponencial":
    rho = rho0 * np.exp(alpha * x)     # [kg/m]
    T   = T0   * np.exp(alpha * x)     # [N]

elif modelo == "catenaria":
    g   = 9.8                           # [m/s²]
    rho = rho0 * np.ones_like(x)       # [kg/m] uniforme
    T   = T0 * np.cosh(rho0 * g * x / T0)  # [N]

# Velocidad local v(x) = sqrt(T/ρ)
v    = np.sqrt(T / rho)
vmax = np.max(v)
dt   = 0.4 * dx / vmax                 # paso temporal conservador (CFL local)

print(f"Modelo: {modelo}")
print(f"v_max = {vmax:.4f} m/s,  dt = {dt:.6f} s")

# =============================================================================
# CONDICIONES INICIALES — pulso gaussiano
# =============================================================================
y     = np.exp(-200 * (x - 0.5)**2)   # pulso angosto centrado en x=0.5
y_old = y.copy()
y_new = np.zeros(Nx)

# =============================================================================
# ALGORITMO CENTRAL — diferencias finitas con T variable
# =============================================================================
Nt   = 2000
pasos_guardar = {0, 200, 500, 1000, 1800}
instantaneas  = {}

for n in range(Nt):
    for i in range(1, Nx - 1):
        # tensión interpolada en los semipuntos
        T_ip = 0.5 * (T[i] + T[i + 1])    # T_{i+1/2}
        T_im = 0.5 * (T[i] + T[i - 1])    # T_{i-1/2}

        # laplaciano con tensión variable
        lap = (T_ip * (y[i + 1] - y[i]) - T_im * (y[i] - y[i - 1])) / dx**2

        # actualización temporal con amortiguamiento opcional
        y_new[i] = (
            2 * y[i]
            - y_old[i]
            + dt**2 * lap / rho[i]
            - kappa * dt * (y[i] - y_old[i])
        )

    # condiciones de frontera Dirichlet
    y_new[0]  = 0.0
    y_new[-1] = 0.0

    # rotar arreglos
    y_old[:] = y
    y[:]     = y_new

    if n in pasos_guardar:
        instantaneas[n] = y.copy()

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: propagación de la onda
ax = axes[0]
for paso, y_snap in instantaneas.items():
    ax.plot(x, y_snap, label=f'n = {paso}')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title(f'Onda con {modelo} — T(x), ρ(x) variables')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel derecho: perfil de velocidad local
ax2 = axes[1]
ax2.plot(x, v, color='darkorange')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('v(x) [m/s]')
ax2.set_title('Velocidad local v(x) = √(T/ρ)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
