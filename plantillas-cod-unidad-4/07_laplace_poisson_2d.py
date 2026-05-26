"""
PLANTILLA 07 — Ecuaciones de Laplace y Poisson 2D
==================================================
Ecuación de Laplace (sin fuentes):
    ∇²V = 0   →   ∂²V/∂x² + ∂²V/∂y² = 0

Ecuación de Poisson (con densidad de carga ρ):
    ∇²V = -ρ/ε₀

Métodos iterativos (sin scipy):

--- Jacobi ---
    V_nuevo[i,j] = 0.25*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
    (+ corrección Poisson:  + 0.25*dx²*ρ[i,j]/ε₀ )
    Actualiza V_nuevo por separado; lento pero sencillo.

--- Gauss-Seidel (in-place, más rápido) ---
    V[i,j] = 0.25*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
    Usa los valores actualizados inmediatamente; converge ~2× más rápido.

--- SOR (Gauss-Seidel + relajación ω) ---
    V_nuevo[i,j] = Gauss-Seidel value
    V[i,j] = V[i,j] + ω*(V_nuevo[i,j] - V[i,j])
    ω ∈ (1, 2): ω=1 → Gauss-Seidel, ω→2 → más rápido (típico ω≈1.9)

Convergencia: max|V_nuevo - V| < tolerancia

Fuente: clase_14-05-26_15-05-26.ipynb, referencias/cap04_ecuaciones_onda_fluidos.md
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS DEL DOMINIO
# =============================================================================
N   = 50        # nodos por lado (dominio cuadrado N×N)
dx  = 1.0       # [m] paso de grilla (dx = dy)
eps0 = 1.0      # permitividad (usar 8.854e-12 para SI; aquí adimensional)

# =============================================================================
# CONDICIONES DE FRONTERA — LAPLACE: capacitor de placas paralelas
# =============================================================================
# Caja externa aterrizada (V=0 en los 4 bordes)
# Placa superior (y alta) a +V_top, placa inferior (y baja) a -V_bot
V_top =  100.0   # [V]
V_bot = -100.0   # [V]
ancho_placa  = N // 2   # nodos de ancho de las placas
centrado     = N // 4   # offset lateral de las placas

# Crear grilla de potencial
V = np.zeros((N, N))

# Condiciones de frontera: bordes aterrorizados
V[0,  :] = 0.0
V[-1, :] = 0.0
V[:,  0] = 0.0
V[:, -1] = 0.0

# Placas del capacitor (condición interior fija)
fila_top = int(0.7 * N)
fila_bot = int(0.3 * N)
col_ini  = N // 4
col_fin  = col_ini + ancho_placa
placa_top = (slice(col_ini, col_fin), fila_top)
placa_bot = (slice(col_ini, col_fin), fila_bot)

V[placa_top] = V_top
V[placa_bot] = V_bot

# =============================================================================
# DENSIDAD DE CARGA ρ (solo para Poisson; poner a cero para Laplace)
# =============================================================================
rho = np.zeros((N, N))
# Ejemplo: carga puntual positiva en (30,30), negativa en (60,30) (Poisson)
# rho[30, 30] = +100.0
# rho[60, 30] = -100.0

# =============================================================================
# ITERACIÓN — GAUSS-SEIDEL + SOR
# =============================================================================
omega       = 1.6         # parámetro SOR (1 → G-S, ~1.9 → más rápido)
tolerancia  = 1e-4
max_iter    = 10000

for it in range(max_iter):
    V_old_max  = V.copy()

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # calcular valor de Gauss-Seidel (más término de Poisson)
            V_gs = 0.25 * (
                V[i + 1, j] + V[i - 1, j]
                + V[i, j + 1] + V[i, j - 1]
                + dx**2 * rho[i, j] / eps0        # = 0 para Laplace puro
            )
            # aplicar SOR
            V[i, j] = V[i, j] + omega * (V_gs - V[i, j])

    # reimponer las placas del capacitor (condiciones interiores fijas)
    V[placa_top] = V_top
    V[placa_bot] = V_bot

    # criterio de convergencia
    error = np.max(np.abs(V - V_old_max))
    if error < tolerancia:
        print(f"Convergió en {it + 1} iteraciones (error = {error:.2e})")
        break
else:
    print(f"No convergió en {max_iter} iteraciones (error final = {error:.2e})")

# =============================================================================
# CAMPO ELÉCTRICO: E = -∇V  (diferencias finitas centradas)
# =============================================================================
Ex = np.zeros_like(V)
Ey = np.zeros_like(V)
Ex[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dx)   # ∂V/∂x → Ex = -∂V/∂x
Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dx)   # ∂V/∂y → Ey = -∂V/∂y

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
x_vals = np.arange(N) * dx
y_vals = np.arange(N) * dx
X, Y   = np.meshgrid(x_vals, y_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: equipotenciales
ax = axes[0]
cf = ax.contourf(X, Y, V.T, levels=40, cmap='coolwarm')
plt.colorbar(cf, ax=ax, label='V [V]')
ax.contour(X, Y, V.T, levels=20, colors='black', linewidths=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Líneas equipotenciales — Laplace/Poisson')

# Panel derecho: campo eléctrico (quiver)
ax2 = axes[1]
paso_flecha = max(1, N // 15)
ax2.quiver(
    X[::paso_flecha, ::paso_flecha],
    Y[::paso_flecha, ::paso_flecha],
    Ex.T[::paso_flecha, ::paso_flecha],
    Ey.T[::paso_flecha, ::paso_flecha],
    color='darkblue', scale=500
)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Campo eléctrico E = −∇V')

plt.tight_layout()
plt.show()
