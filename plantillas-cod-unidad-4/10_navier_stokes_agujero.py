"""
PLANTILLA 10 — Navier-Stokes 2D con geometría de boquilla / agujero
====================================================================
Problema: flujo a través de un agujero en la pared inferior del dominio.
La boquilla introduce velocidades verticales (vy < 0 hacia abajo).

Campos:
  - ψ: función de corriente
  - ω: vorticidad

Condiciones de contorno especiales:
  - BelowHole:   debajo del agujero, vy calculada por conservación de energía
                  vy = -sqrt(2*g*h*(Ny + Nb - j))
  - BorderRight: salida lateral
  - BottomBefore: pared inferior sólida antes del agujero
  - Top:         tapa superior libre (ω=0)
  - Left:        pared izquierda con condición de Neumann en ω

Parámetro R:  R = V₀*h/ν  (Reynolds de la grilla)

Fuente: clase_08-05-26.py  |  Landau Listing 4.10 (Torricelli.py)
Algoritmo: ψ y ω con SOR; condiciones de frontera copiadas del Listing 4.10.
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================
Nx, Ny = 60, 60    # nodos del dominio
h      = 0.4       # [m] paso de grilla
V0     = 8e-4      # [m/s] velocidad de referencia
g      = 9.8       # [m/s²] gravedad
omega  = 0.1       # parámetro SOR (conservador para estabilidad)
nu     = 0.5       # [m²/s] viscosidad cinemática
R      = V0 * h / nu   # número de Reynolds de grilla
Niter  = 4000
Nb     = 15        # nodos antes del agujero en x
Ndown  = 20        # nodos hacia abajo en la región del agujero

print(f"R = {R:.5f}")

# =============================================================================
# INICIALIZACIÓN
# =============================================================================
psi = np.zeros((Nx + 1, Ny + 1))
w   = np.zeros((Nx + 1, Ny + 1))

# =============================================================================
# CONDICIONES DE FRONTERA
# =============================================================================
def BelowHole():
    """Debajo del agujero: velocidad impuesta por conservación de energía."""
    for i in range(Nb + 1, Nx + 1):
        psi[i, 0] = psi[i - 1, 1]
        w[i - 1, 0] = w[i - 1, 1]
        for j in range(0, Ndown + 1):
            if i == Nb:
                vy = 0.0
            elif i == Nx:
                vy = -np.sqrt(2 * g * h * (Ny + Nb - j))
            elif i == Nx - 1:
                vy = -np.sqrt(2 * g * h * (Ny + Nb - j)) / 2
            else:
                vy = 0.0
            psi[i, j] = psi[i - 1, j] - vy * h

def BorderRight():
    """Salida lateral derecha."""
    for j in range(1, Ny + 1):
        vy = -np.sqrt(2 * g * h * (Ny - 1))
        psi[Nx, j] = psi[Nx - 1, j] + vy * h
        psi[Nx, j] = psi[Nx, j - 1]
        w[Nx, j]   = -2 * (psi[Nx, j] - psi[Nx, j - 1]) / h**2

def BottomBefore():
    """Pared inferior sólida antes del agujero."""
    for i in range(1, Nb + 1):
        psi[i, Ndown] = psi[i, Ndown - 1]
        w[i, Ndown]   = -2 * (psi[i, 0] - psi[i, 1]) / h**2

def Top():
    """Tapa superior libre (vorticidad cero)."""
    for i in range(1, Nx):
        psi[i, Ny] = psi[i, Ny - 1]
        w[i, Ny]   = 0.0

def Left():
    """Pared izquierda: condición de Neumann."""
    for j in range(Ndown, Ny):
        w[0, j]   = -2 * (psi[0, j] - psi[1, j]) / h**2
        psi[0, j] = psi[1, j]

def Borders():
    BelowHole()
    BorderRight()
    BottomBefore()
    Top()
    Left()

# =============================================================================
# ITERACIÓN SOR
# =============================================================================
def relajar():
    Borders()

    # --- ecuación de Poisson para ψ ---
    for i in range(1, Nx):
        for j in range(1, Ny):
            r1 = omega * (
                (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] - h**2 * w[i,j])
                * 0.25 - psi[i, j]
            )
            psi[i, j] += r1

    Borders()

    # --- ecuación de transporte de vorticidad ---
    for i in range(1, Nx):
        for j in range(1, Ny):
            a1 = w[i+1,j] + w[i-1,j] + w[i,j+1] + w[i,j-1]
            # términos no lineales (convección de vorticidad)
            a2 = (psi[i, j+1] - psi[i, j-1]) * (w[i+1, j] - w[i-1, j])
            a3 = (psi[i+1, j] - psi[i-1, j]) * (w[i, j+1] - w[i, j-1])
            r2 = omega * (
                (a1 + (R / 4.0) * (a3 - a2)) / 4.0 - w[i, j]
            )
            w[i, j] += r2

# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================
for it in range(Niter):
    relajar()
    if it % 500 == 0:
        print(f"  it = {it}")

# =============================================================================
# VELOCIDADES: vx = ∂ψ/∂y,  vy = -∂ψ/∂x
# =============================================================================
vx = np.zeros_like(psi)
vy = np.zeros_like(psi)
for i in range(1, Nx):
    for j in range(1, Ny):
        vx[i, j] =  (psi[i, j+1] - psi[i, j-1]) / (2 * h)
        vy[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * h)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
x = np.arange(Nx + 1) * h
y = np.arange(Ny + 1) * h
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Función de corriente
ax = axes[0]
ax.imshow(psi.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto', cmap='viridis')
ax.set_title('Función de corriente ψ')
ax.set_xlabel('x'); ax.set_ylabel('y')

# Vorticidad
ax = axes[1]
ax.imshow(w.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto', cmap='RdBu_r')
ax.set_title('Vorticidad ω')
ax.set_xlabel('x'); ax.set_ylabel('y')

# Líneas de corriente
ax = axes[2]
speed = np.sqrt(vx.T**2 + vy.T**2)
ax.streamplot(X, Y, vx.T, vy.T, density=1.5, color=speed, cmap='plasma')
ax.set_title('Campo de velocidades')
ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle(f'N-S boquilla  (R={R:.3f}, ω={omega})', fontsize=12)
plt.tight_layout()
plt.show()
