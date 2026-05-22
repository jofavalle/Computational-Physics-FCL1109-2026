"""
PLANTILLA 04 — Membrana vibrante 2D
=====================================
Ecuación diferencial:
    ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)

Discretización en 2D (diferencias finitas, grilla NxN):
    u_nuevo[i,j] = 2*u[i,j] - u_prev[i,j]
                   + r² * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
    donde r = c*dt/dx  (debe ser ≤ 1/√2 en 2D para estabilidad)

⚠ CONDICIÓN CFL EN 2D:
    r = c*dt/dx ≤ 1/√2 ≈ 0.707

Condiciones de frontera: Dirichlet u=0 en todos los bordes (membrana fija).

Fuente: clase_23-04-26.py  |  Landau Listing 4.4 (Waves2D.py)
Algoritmo: leapfrog 2D con vecinos i±1, j±1; u[i,j,2]↔u_next, [1]↔u, [0]↔u_prev.
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
c  = 1.0   # [m/s] velocidad de propagación en la membrana
L  = 1.0   # [m]   tamaño del dominio (membrana cuadrada L×L)

# =============================================================================
# GRILLA 2D
# =============================================================================
N  = 71                         # puntos por lado
dx = L / N                      # [m] paso espacial (dx = dy)
dt = 0.001                      # [s] paso temporal (ajustar para CFL)

# Número de Courant 2D
r  = c * dt / dx
assert r <= 1.0 / np.sqrt(2), (
    f"¡Inestable en 2D! r = {r:.4f} > 1/√2 = {1/np.sqrt(2):.4f}. Reduce dt."
)
print(f"r = {r:.4f}  (CFL 2D estable ✓, límite = {1/np.sqrt(2):.4f})")

# =============================================================================
# CONDICIONES INICIALES — pulso gaussiano centrado
# =============================================================================
u      = np.zeros((N, N))
u_prev = np.zeros((N, N))
u_next = np.zeros((N, N))

# Construir pulso gaussiano sobre la grilla 2D
for i in range(N):
    for j in range(N):
        xi = i * dx
        yj = j * dx
        u[i, j] = np.exp(-100 * ((xi - 0.5)**2 + (yj - 0.5)**2))

# --- Opción con numpy (más rápido, descomentar si la versión anterior es lenta) ---
# ix  = np.arange(N) * dx
# Xi, Yj = np.meshgrid(ix, ix, indexing='ij')
# u = np.exp(-100 * ((Xi - 0.5)**2 + (Yj - 0.5)**2))

u_prev = u.copy()

# =============================================================================
# ALGORITMO CENTRAL — paso temporal 2D
# =============================================================================
Nt              = 300
pasos_visualizar = [0, 50, 150, 299]
instantaneas    = {}

for n in range(Nt):
    # --- nodos interiores: diferencias finitas 2D ---
    u_next[1:-1, 1:-1] = (
        2 * u[1:-1, 1:-1]
        - u_prev[1:-1, 1:-1]
        + r**2 * (
            u[2:,   1:-1]    # vecino i+1
            + u[:-2,  1:-1]  # vecino i-1
            + u[1:-1, 2:]    # vecino j+1
            + u[1:-1, :-2]   # vecino j-1
            - 4 * u[1:-1, 1:-1]
        )
    )

    # --- condiciones de frontera Dirichlet: u=0 en todos los bordes ---
    u_next[0,  :] = 0.0
    u_next[-1, :] = 0.0
    u_next[:,  0] = 0.0
    u_next[:, -1] = 0.0

    # --- rotar arreglos ---
    u_prev = u.copy()
    u      = u_next.copy()

    if n in pasos_visualizar:
        instantaneas[n] = u.copy()

# =============================================================================
# VISUALIZACIÓN — mapas de calor en 4 instantes
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for idx, (paso, u_snap) in enumerate(instantaneas.items()):
    ax = axes.flat[idx]
    im = ax.imshow(
        u_snap.T,
        origin='lower',
        extent=[0, L, 0, L],
        cmap='viridis',
        vmin=-0.5, vmax=1.0
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f't = {paso * dt:.4f} s  (paso {paso})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

fig.suptitle(f'Membrana vibrante 2D  (c={c} m/s, N={N}×{N}, r={r:.3f})', fontsize=12)
plt.tight_layout()
plt.show()
