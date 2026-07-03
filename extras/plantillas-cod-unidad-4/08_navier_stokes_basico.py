"""
PLANTILLA 08 — Navier-Stokes 2D: función de corriente ψ y vorticidad ω
========================================================================
Las ecuaciones de Navier-Stokes incompresibles en 2D se reformulan usando:
  - ψ(x,y): función de corriente (streamfunction)
  - ω(x,y): vorticidad = ∂vy/∂x - ∂vx/∂y

Relaciones:
    vx =  ∂ψ/∂y
    vy = -∂ψ/∂x

Sistema de dos ecuaciones:
    ∇²ψ = -ω                                    ← Poisson para ψ
    ∂ω/∂t = -∂(ψ,ω)/∂(x,y) + (1/R) ∇²ω        ← transporte de vorticidad

En estado estacionario (∂ω/∂t = 0), el sistema se resuelve iterativamente
mediante SOR (Successive Over-Relaxation):

    ψ_nuevo[i,j] = 0.25*(ψ[i+1,j] + ψ[i-1,j] + ψ[i,j+1] + ψ[i,j-1] + h²*ω[i,j])
    ω_nuevo[i,j] = 0.25*(ω[i+1,j]+ω[i-1,j]+ω[i,j+1]+ω[i,j-1])
                   + (R/16)*(ψ[i,j+1]-ψ[i,j-1])*(ω[i+1,j]-ω[i-1,j])
                   - (R/16)*(ψ[i+1,j]-ψ[i-1,j])*(ω[i,j+1]-ω[i,j-1])

Número de Reynolds:
    R = V₀ * h / ν   (donde V₀ es la velocidad de referencia)

Problema: flujo en un canal con tapa superior deslizante (lid-driven cavity)
Fuente: clase_05-05-26.py  |  Landau Listing 4.9 (Beam.py) — versión simplificada sin viga
Algoritmo SOR: r1 = ω*((Σψ + h²·w)/4 − ψ)  equivale al relax() de Landau.
Nota: clase_05-05-26 usa r1 = ω*(Σu − 4u + h²w) sin el /4; es el mismo SOR con ω_eff 4×.
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS DEL PROBLEMA
# =============================================================================
Nx, Ny = 400, 400   # nodos en x e y
h      = 1.0        # [m] paso de grilla (dx = dy = h)
V0     = 1.0        # [m/s] velocidad de la tapa superior
nu     = 1.0        # [m²/s] viscosidad cinemática
omega  = 0.1        # parámetro SOR (0 < ω ≤ 2, empezar conservador)
R      = V0 * h / nu  # número de Reynolds (aquí efectivo por grilla)
Niter  = 100        # iteraciones (aumentar para mejor convergencia)

print(f"Reynolds efectivo R = {R:.3f}")

# =============================================================================
# INICIALIZACIÓN DE CAMPOS
# =============================================================================
psi = np.zeros((Nx + 1, Ny + 1))   # función de corriente ψ
w   = np.zeros((Nx + 1, Ny + 1))   # vorticidad ω

# =============================================================================
# CONDICIONES DE FRONTERA — Lid-driven cavity
# =============================================================================
def aplicar_frontera():
    # Tapa superior (y = Ny): se mueve a velocidad V0 en x
    # ψ aumenta linealmente: ψ[i, Ny] = ψ[i, Ny-1] + h*V0
    for i in range(1, Nx):
        psi[i, Ny] = psi[i, Ny - 1] + h * V0
        w[i,   Ny] = 0.0                          # vorticidad libre en la tapa

    # Entrada (x = 0): flujo paralelo
    for j in range(1, Ny):
        psi[0, j] = psi[1, j]
        w[0,   j] = 0.0

    # Salida (x = Nx): condición de salida (extrapolación)
    for j in range(1, Ny):
        psi[Nx, j] = psi[Nx - 1, j]
        w[Nx,   j] = w[Nx - 1,   j]

# =============================================================================
# ITERACIÓN SOR
# =============================================================================
def relajar():
    for i in range(1, Nx):
        for j in range(1, Ny):
            # --- ecuación de Poisson para ψ: ∇²ψ = -ω ---
            r1 = omega * (
                (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
                 + h**2 * w[i, j]) * 0.25
                - psi[i, j]
            )
            psi[i, j] += r1

            # --- ecuación de vorticidad estacionaria ---
            a1 = w[i+1, j] + w[i-1, j] + w[i, j+1] + w[i, j-1]
            w_nuevo = 0.25 * a1
            rw = w_nuevo - w[i, j]
            w[i, j] += omega * rw

aplicar_frontera()
for it in range(Niter):
    relajar()
    aplicar_frontera()
    if it % 10 == 0:
        print(f"  iteración {it}")

# =============================================================================
# VELOCIDADES A PARTIR DE ψ:  vx = ∂ψ/∂y,  vy = -∂ψ/∂x
# =============================================================================
vx = np.zeros_like(psi)
vy = np.zeros_like(psi)
vx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
vy[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * h)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
X, Y = np.meshgrid(np.arange(Nx + 1) * h, np.arange(Ny + 1) * h)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Función de corriente ψ (líneas de corriente = curvas de nivel de ψ)
ax = axes[0]
cf = ax.contourf(X, Y, psi.T, levels=40, cmap='viridis')
plt.colorbar(cf, ax=ax, label='ψ(x,y)')
ax.set_title('Función de corriente ψ')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Líneas de corriente (streamplot)
ax2 = axes[1]
speed = np.sqrt(vx.T**2 + vy.T**2)
ax2.streamplot(X, Y, vx.T, vy.T, color=speed, cmap='plasma', density=1.5)
ax2.set_title('Líneas de corriente + velocidad')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.suptitle(f'N-S: lid-driven cavity  (Niter={Niter})', fontsize=12)
plt.tight_layout()
plt.show()
