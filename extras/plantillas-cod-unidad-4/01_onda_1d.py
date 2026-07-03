"""
PLANTILLA 01 - Ecuación de onda 1D con diferencias finitas
===========================================================
Ecuación diferencial:
    ∂²y/∂t² = c² ∂²y/∂x²

Discretización (diferencias finitas centradas en espacio y tiempo):
    y[i, n+1] = 2*y[i,n] - y[i,n-1] + r²*(y[i+1,n] - 2*y[i,n] + y[i-1,n])
    donde r = c*dt/dx  (número de Courant)

⚠ CONDICIÓN DE ESTABILIDAD (CFL):  r = c*dt/dx ≤ 1
  Si r > 1 la solución diverge exponencialmente.

Con amortiguamiento (fricción viscosa):
    y[i, n+1] = (1/(1+κ*dt)) * [2*y[i,n]*(1+κ*dt) - y[i,n-1]
                + r²*(y[i+1,n] - 2*y[i,n] + y[i-1,n])]
    - o equivalentemente, ver la sección de amortiguamiento abajo -

Fuente: clase_14-04-26.ipynb, clase_16-04-26.ipynb  |  Landau Listing 4.1 (EqStringMovMat.py)
Algoritmo: leapfrog idéntico al de Landau; xi[i,2] ↔ y_nuevo, xi[i,1] ↔ y_actual, xi[i,0] ↔ y_anterior.
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L   = 1.0   # [m]     longitud de la cuerda
c   = 1.0   # [m/s]   velocidad de propagación
kappa = 0.0 # [1/s]   coeficiente de amortiguamiento (0 = sin fricción)

# =============================================================================
# GRILLA ESPACIAL Y TEMPORAL
# =============================================================================
Nx = 100                      # número de puntos en x
dx = L / (Nx - 1)             # [m] paso espacial
dt = 0.8 * dx / c             # [s] paso temporal (factor 0.8 garantiza CFL < 1)
Nt = 500                      # número de pasos temporales

# Verificación de estabilidad CFL
r = c * dt / dx
assert r <= 1.0, f"¡Inestable! r = {r:.4f} > 1. Reduce dt o aumenta dx."
print(f"r = {r:.4f}  (CFL estable ✓)")

x = np.linspace(0, L, Nx)    # grilla espacial

# =============================================================================
# CONDICIONES INICIALES
# =============================================================================
# --- Opción A: pulso gaussiano centrado ---
sigma = L / 10
x0    = L / 2
y_actual  = np.exp(-((x - x0)**2) / (2 * sigma**2))

# --- Opción B: cuerda pulsada triangular (descomentar) ---
# y_actual = np.where(x < L/2, 2*x/L, 2*(L-x)/L)

# --- Opción C: modo normal n=1 (descomentar) ---
# y_actual = np.sin(np.pi * x / L)

# El paso n-1 se inicializa igual al paso n (velocidad inicial = 0)
y_anterior = y_actual.copy()
y_nuevo    = np.zeros(Nx)

# =============================================================================
# CONDICIONES DE FRONTERA
# =============================================================================
# Dirichlet (extremos fijos): y[0] = y[-1] = 0
# Si se quieren extremos libres (Neumann): y[0] = y[1], y[-1] = y[-2]

def aplicar_frontera(y):
    y[0]  = 0.0   # extremo izquierdo fijo
    y[-1] = 0.0   # extremo derecho fijo
    return y

# =============================================================================
# INTEGRACIÓN TEMPORAL - ALGORITMO CENTRAL
# =============================================================================
# Guardamos algunas instantáneas para graficar
pasos_guardar = [0, 50, 100, 200, 400]
instantaneas  = {}

for n in range(Nt):
    # --- nodos interiores: fórmula de diferencias finitas de segundo orden ---
    y_nuevo[1:-1] = (
        2 * y_actual[1:-1]
        - y_anterior[1:-1]
        + r**2 * (y_actual[2:] - 2 * y_actual[1:-1] + y_actual[:-2])
    )

    # --- amortiguamiento (solo si kappa > 0) ---
    if kappa > 0.0:
        # Corrección por fricción: divide entre (1 + kappa*dt)
        y_nuevo[1:-1] = (
            y_nuevo[1:-1] + kappa * dt * y_anterior[1:-1]
        ) / (1.0 + kappa * dt)

    # --- condiciones de frontera ---
    y_nuevo = aplicar_frontera(y_nuevo)

    # --- avanzar el tiempo: rotar los arreglos ---
    y_anterior[:] = y_actual
    y_actual[:]   = y_nuevo

    # --- guardar instantánea si corresponde ---
    if n in pasos_guardar:
        instantaneas[n] = y_actual.copy()

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for paso, y_snap in instantaneas.items():
    t_snap = paso * dt
    ax.plot(x, y_snap, label=f't = {t_snap:.3f} s')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title(f'Ecuación de onda 1D  (c = {c} m/s, r = {r:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
