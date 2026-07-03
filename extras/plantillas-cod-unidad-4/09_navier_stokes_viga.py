"""
PLANTILLA 09 - Navier-Stokes 2D con obstáculo rectangular (viga)
=================================================================
Misma formulación ψ-ω que la plantilla 08, pero con una viga rectangular
dentro del dominio. Las celdas interiores de la viga se saltean en el loop.

Geometría:
  - Canal 2D: Nx×Ny nodos
  - Viga: desde x0 hasta x1 en x, desde y0 hasta y1 en y
  - Condición de contorno en la viga: ψ constante (flujo no penetra)

Ecuaciones de frontera en la viga:
  - ψ en la viga = ψ[x0, y0]  (superficie sólida: línea de corriente)
  - ω en la superficie de la viga: calculada desde el Laplaciano de ψ

Fuente: clase_07-05-26.py  |  Landau Listing 4.9 (Beam.py)
⚠ El signo del término convectivo sigue a Landau Listing 4.9 y clase_08-05-26.py,
  no a clase_07-05-26.py (que tiene un error de signo en ese término).
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================
Nx, Ny  = 70, 24       # nodos del canal
h       = 1.0          # [m] paso de grilla
V0      = 1.0          # [m/s] velocidad de entrada
omega   = 0.3          # parámetro SOR
R       = 0.1          # número de Reynolds efectivo
tol     = 1e-3         # tolerancia de convergencia
max_iter = 5000

# Geometría de la viga
L_viga  = 8.0          # [m] longitud de la viga
H_viga  = 4.0          # [m] altura de la viga
x0_v = 20                           # columna de inicio
x1_v = x0_v + int(L_viga / h)      # columna de fin
y0_v = Ny // 2 - int(H_viga / (2 * h))   # fila inferior
y1_v = y0_v + int(H_viga / h)      # fila superior

print(f"Viga: columnas [{x0_v}, {x1_v}), filas [{y0_v}, {y1_v})")

# =============================================================================
# INICIALIZACIÓN
# =============================================================================
psi = np.zeros((Nx, Ny))
w   = np.zeros((Nx, Ny))

# =============================================================================
# CONDICIONES DE FRONTERA
# =============================================================================
def aplicar_frontera():
    # Entrada: perfil lineal (flujo uniforme)
    for j in range(Ny):
        psi[0, j] = V0 * j

    # Salida: extrapolación
    psi[-1, :] = psi[-2, :]
    w[-1,   :] = w[-2,   :]

    # Tapa superior: flujo uniforme
    psi[:, -1] = V0 * (Ny - 1)

    # Pared inferior: no deslizamiento
    psi[:, 0] = 0.0

    # Viga: ψ constante = valor en la esquina de la viga
    psi[x0_v:x1_v, y0_v:y1_v] = psi[x0_v, y0_v]

# =============================================================================
# ITERACIÓN SOR CON VIGA
# =============================================================================
def relajar():
    max_res = 0.0
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # saltar el interior de la viga
            if x0_v <= i < x1_v and y0_v <= j < y1_v:
                continue

            # --- ψ: ecuación de Poisson ∇²ψ = -ω ---
            psi_nuevo = 0.25 * (
                psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
                + h**2 * w[i, j]
            )
            ru = psi_nuevo - psi[i, j]
            psi[i, j] += omega * ru

            # --- ω: ecuación de transporte de vorticidad ---
            a1 = w[i+1,j] + w[i-1,j] + w[i,j+1] + w[i,j-1]

            # término convectivo - signo según Landau Listing 4.9 y clase_08-05-26.py
            # de ∂ω/∂t + ψ_y·∂ω/∂x − ψ_x·∂ω/∂y = ν·∇²ω se obtiene:
            # ω_GS = (Σω − (R/4)·(a2 − a3)) / 4  ≡  (Σω + (R/4)·(a3 − a2)) / 4
            a2 = (psi[i, j+1] - psi[i, j-1]) * (w[i+1, j] - w[i-1, j])  # ψ_y · ω_x (×4h²)
            a3 = (psi[i+1, j] - psi[i-1, j]) * (w[i, j+1] - w[i, j-1])  # ψ_x · ω_y (×4h²)
            w_nuevo = 0.25 * (a1 + (R / 4.0) * (a3 - a2))
            rw = w_nuevo - w[i, j]
            w[i, j] += omega * rw

            max_res = max(max_res, abs(ru), abs(rw))

    return max_res

# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================
for it in range(max_iter):
    aplicar_frontera()
    residual = relajar()

    if it % 500 == 0:
        print(f"  it = {it:5d}  residual = {residual:.5f}")

    if residual < tol:
        print(f"Convergencia en {it} iteraciones (residual = {residual:.2e})")
        break

# =============================================================================
# VELOCIDADES: vx = ∂ψ/∂y,  vy = -∂ψ/∂x
# =============================================================================
vx = np.zeros_like(psi)
vy = np.zeros_like(psi)
vx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
vy[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * h)

# Enmascarar la viga para la visualización
mascara_viga = np.zeros((Nx, Ny), dtype=bool)
mascara_viga[x0_v:x1_v, y0_v:y1_v] = True
psi_vis = np.where(mascara_viga, np.nan, psi)
w_vis   = np.where(mascara_viga, np.nan, w)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
X, Y = np.meshgrid(np.arange(Nx) * h, np.arange(Ny) * h)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Función de corriente
ax = axes[0, 0]
cf = ax.contourf(X, Y, psi_vis.T, levels=40, cmap='viridis')
plt.colorbar(cf, ax=ax, label='ψ')
ax.set_title('Función de corriente ψ')
ax.set_xlabel('x'); ax.set_ylabel('y')

# Vorticidad
ax = axes[0, 1]
cf2 = ax.contourf(X, Y, w_vis.T, levels=40, cmap='RdBu_r')
plt.colorbar(cf2, ax=ax, label='ω')
ax.set_title('Vorticidad ω')
ax.set_xlabel('x'); ax.set_ylabel('y')

# Líneas de corriente
ax = axes[1, 0]
speed = np.sqrt(vx.T**2 + vy.T**2)
ax.streamplot(X, Y, vx.T, vy.T, color=speed, cmap='plasma', density=1.5)
ax.add_patch(plt.Rectangle(
    (x0_v * h, y0_v * h), L_viga, H_viga, color='gray', zorder=5
))
ax.set_title('Líneas de corriente')
ax.set_xlabel('x'); ax.set_ylabel('y')

# Campo de vectores velocidad
ax = axes[1, 1]
paso = max(1, Nx // 15)
ax.quiver(X[::1, ::paso], Y[::1, ::paso], vx.T[::1, ::paso], vy.T[::1, ::paso])
ax.add_patch(plt.Rectangle(
    (x0_v * h, y0_v * h), L_viga, H_viga, color='gray', zorder=5
))
ax.set_title('Campo de velocidades')
ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle(f'N-S con viga  (R={R}, ω={omega})', fontsize=12)
plt.tight_layout()
plt.show()
