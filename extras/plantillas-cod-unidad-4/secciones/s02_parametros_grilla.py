"""
SECCIÓN: PARÁMETROS FÍSICOS Y CONSTRUCCIÓN DE GRILLA
=====================================================
Patrón que aparece al inicio de TODOS los scripts de la unidad 4.
Estructura: constantes → grilla → verificación de estabilidad.

La verificación SIEMPRE va con assert o if/raise.
"""

import numpy as np

# =============================================================================
# PATRÓN 1 — Onda 1D (ecuación de onda, leapfrog)
# =============================================================================
L  = 1.0    # [m]   longitud del dominio
c  = 1.0    # [m/s] velocidad de propagación
Nx = 200    # número de nodos
dx = L / (Nx - 1)       # L/(Nx-1) → incluye los dos extremos
dt = 0.8 * dx / c       # CFL = 0.8 (seguro)
Nt = 500                # pasos temporales
r  = c * dt / dx        # número de Courant (debe ser ≤ 1)

assert r <= 1.0, f"Inestable: r = {r:.4f} > 1"

x = np.linspace(0, L, Nx)      # [0, L] con Nx puntos


# =============================================================================
# PATRÓN 2 — Onda con propiedades variables T(x), ρ(x)
# =============================================================================
# (dt depende de la velocidad MÁXIMA local)
vmax = np.max(np.sqrt(T / rho))     # calcular después de definir T, rho
dt   = 0.4 * dx / vmax             # factor conservador


# =============================================================================
# PATRÓN 3 — Calor/difusión FTCS
# =============================================================================
alpha = 0.01   # [m²/s] difusividad térmica
Nx    = 100
dx    = L / (Nx - 1)
r_cal = 0.4                       # elegir r < 0.5 para Von Neumann
dt    = r_cal * dx**2 / alpha     # despejar dt a partir del r deseado
Nt    = 3000

assert alpha * dt / dx**2 <= 0.5, "Inestable FTCS"

x = np.linspace(0, L, Nx)


# =============================================================================
# PATRÓN 4 — Grilla 2D (membrana vibrante, Laplace, N-S)
# =============================================================================
Nx, Ny = 71, 71          # nodos por dimensión
L      = 1.0             # [m] tamaño del dominio cuadrado
dx     = L / Nx          # paso espacial (igual en x e y)
dt     = 0.001           # paso temporal
r_2d   = c * dt / dx
assert r_2d <= 1.0 / np.sqrt(2), f"Inestable 2D: r = {r_2d:.4f}"

# Grilla de coordenadas (para visualización o condición inicial vectorizada)
x  = np.arange(Nx) * dx
y  = np.arange(Ny) * dx
X, Y = np.meshgrid(x, y, indexing='ij')  # X[i,j] = x_i,  Y[i,j] = y_j

# Campo 2D
u      = np.zeros((Nx, Ny))
u_prev = np.zeros((Nx, Ny))
u_next = np.zeros((Nx, Ny))


# =============================================================================
# PATRÓN 5 — Navier-Stokes (ψ-ω)
# =============================================================================
Nx, Ny = 70, 24     # nodos del canal
h      = 1.0        # paso de grilla [m]  (usa h en lugar de dx)
V0     = 1.0        # velocidad de referencia [m/s]
nu     = 0.5        # viscosidad cinemática [m²/s]
R      = V0 * h / nu  # número de Reynolds de grilla
omega  = 0.3        # parámetro SOR (iniciar conservador, aumentar si converge bien)

psi = np.zeros((Nx, Ny))    # función de corriente
w   = np.zeros((Nx, Ny))    # vorticidad


# =============================================================================
# NOTAS RÁPIDAS — qué usa linspace vs arange
# =============================================================================
# np.linspace(0, L, N)  → N puntos de 0 a L INCLUSIVE  →  dx = L/(N-1)
# np.arange(N) * dx     → N puntos de 0 a (N-1)*dx     →  dx elegido
#
# Regla: usar linspace cuando los extremos son condiciones de frontera.
#        usar arange cuando el paso dx es el dato natural (como en N-S).
