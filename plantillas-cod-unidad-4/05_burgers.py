"""
PLANTILLA 05 — Ecuación de Burgers (advección-difusión no lineal)
==================================================================
Ecuación diferencial:
    ∂u/∂t = ε ∂²u/∂x² - μ u ∂u/∂x

Discretización explícita (upwind / diferencias centradas):
    u_nuevo[i] = u[i]
                 + ε*(u[i+1] - 2*u[i] + u[i-1])       ← difusión
                 - μ*u[i]*(u[i+1] - u[i-1]) / (2)      ← advección (no lineal)

  (se omite dt y dx aquí porque ε y μ ya absorben esas escalas en el código)

⚠ Esta ecuación puede generar choques (frentes abruptos) cuando μ >> ε.
⚠ Estabilidad: ε ≥ μ*u_max*dx/2  (condición aproximada)

Condiciones iniciales posibles:
  - escalón suavizado:  u = 0.5*(1 - tanh(x/5 - 5))
  - gaussiana:          u = exp(-(x-30)²/50)
  - doble pulso
  - ruido aleatorio     (se verá cómo evoluciona a un frente)

Fuente: clase_24-04-26_animated.py
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================
N   = 300       # número de puntos espaciales
dx  = 0.4       # [m] paso espacial
dt  = 1.0       # [-] paso temporal (adimensional en este modelo)
eps = 0.2       # coeficiente de difusión
mu  = 0.1       # coeficiente de advección (no linealidad)

x = np.arange(N) * dx       # grilla espacial: 0, dx, 2dx, ...

# =============================================================================
# CONDICIONES INICIALES (descomentar la deseada)
# =============================================================================
# --- Opción A: escalón suavizado (onda de choque) ---
u = 0.5 * (1 - np.tanh(x / 5 - 5))

# --- Opción B: pulso gaussiano (descomentar) ---
# u = np.exp(-(x - 30)**2 / 50)

# --- Opción C: doble pulso (descomentar) ---
# u = np.exp(-(x - 20)**2 / 40) + 0.5 * np.exp(-(x - 60)**2 / 40)

# --- Opción D: rectangular (descomentar) ---
# u = np.zeros_like(x); u[60:100] = 1.0

# --- Opción E: ruido aleatorio (descomentar) ---
# u = 0.2 * np.random.rand(N)

u_old = u.copy()
u_new = np.zeros(N)

# =============================================================================
# INTEGRACIÓN TEMPORAL
# =============================================================================
Nt              = 500
pasos_guardar   = {0, 100, 200, 350, 499}
instantaneas    = {0: u.copy()}

for n in range(1, Nt):
    # nodos interiores: difusión + advección no lineal
    u_new[1:-1] = (
        u_old[1:-1]
        + eps * (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2])      # ← difusión
        - mu  * u_old[1:-1] * (u_old[2:] - u_old[:-2])           # ← advección
    )

    # condiciones de frontera periódicas (o Dirichlet: u_new[0]=u_new[-1]=0)
    u_new[0]  = u_new[1]
    u_new[-1] = u_new[-2]

    # rotar
    u_old[:] = u_new

    if n in pasos_guardar:
        instantaneas[n] = u_new.copy()

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for paso, u_snap in instantaneas.items():
    ax.plot(x, u_snap, label=f'paso {paso}  (t={paso*dt:.0f})')

ax.set_xlabel('x [m]')
ax.set_ylabel('u(x, t)')
ax.set_title(f'Ecuación de Burgers  (ε={eps}, μ={mu})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
