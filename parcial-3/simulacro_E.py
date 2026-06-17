import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARCIAL 3 — SIMULACRO E
# Tema  : Electricidad y Magnetismo — Ecuación de Poisson 2D
# Método: Diferencias finitas + SOR (Sobrerelajación Sucesiva)
# Ref.  : clase_14-05-26, clase_28-05-26, Landau Cap. 5
# =============================================================================
#
# SISTEMA: Caja cuadrada conductora aterrizada (V = 0 en paredes) con cuatro
# placas cargadas en su interior, dispuestas en configuración de cuadrupolo:
#
#       ┌──────────────────────────┐
#       │   [+ρ₀]      [−ρ₀]      │  V = 0 (tierra)
#       │                          │
#       │   [−ρ₀]      [+ρ₀]      │
#       └──────────────────────────┘
#
# La ecuación de Poisson para el potencial eléctrico V(x,y) es:
#
#       ∇²V = −ρ(x,y)/ε₀
#
# Discretizada con diferencias finitas centrales (paso Δ):
#
#       V[i,j] = ¼ (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
#                + Δ²·ρ[i,j] / (4·ε₀)
#
# Actualización SOR con factor de sobrerelajación ω ∈ (1, 2):
#
#       V[i,j] ← (1−ω)·V[i,j] + ω·V_GS[i,j]
#
# =============================================================================

# ─── Parámetros del dominio ──────────────────────────────────────────────────
N       = 100          # nodos por lado
delta   = 1.0          # [u.r.] tamaño del paso de la malla
epsilon0 = 1.0         # permitividad (unidades reducidas)
rho0    = 100.0        # [u.r.] densidad de carga de las placas

omega   = 1.85         # factor de sobrerelajación SOR  (ω ∈ (1.5, 2.0))
tol     = 1e-5         # criterio de convergencia (error máximo)
N_iter  = 10000        # máximo de iteraciones

# ─── Distribución de carga: cuatro placas en configuración de cuadrupolo ─────
rho = np.zeros((N, N))

# Coordenadas de las placas (filas, columnas) — índices de nodos
#   Placa 1 (sup. izq.) → carga +ρ₀
rho[15:35, 15:35] = +rho0
#   Placa 2 (sup. der.) → carga −ρ₀
rho[15:35, 65:85] = -rho0
#   Placa 3 (inf. izq.) → carga −ρ₀
rho[65:85, 15:35] = -rho0
#   Placa 4 (inf. der.) → carga +ρ₀
rho[65:85, 65:85] = +rho0

# ─── Condiciones de frontera: caja aterrizada ─────────────────────────────────
V = np.zeros((N, N))
# V en los bordes = 0 (inicialmente ya son cero y no se actualizan)

# ─── Iteración SOR ───────────────────────────────────────────────────────────
historial_error = []

for iteracion in range(N_iter):
    V_old = V.copy()

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # Valor de Gauss-Seidel (Poisson)
            V_GS = 0.25 * (V[i+1, j] + V[i-1, j] +
                           V[i, j+1] + V[i, j-1] +
                           delta**2 * rho[i, j] / epsilon0)
            # Actualización SOR
            V[i, j] = (1.0 - omega) * V[i, j] + omega * V_GS

    error = np.max(np.abs(V - V_old))
    historial_error.append(error)

    if iteracion % 200 == 0:
        print(f"  Iter {iteracion:5d}   error = {error:.3e}")

    if error < tol:
        print(f"\nConvergencia alcanzada en iteración {iteracion}  (error = {error:.2e})")
        break

# ─── Campo eléctrico: E = −∇V (diferencias finitas centrales) ────────────────
Ex = np.zeros_like(V)
Ey = np.zeros_like(V)

Ex[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * delta)   # −∂V/∂x
Ey[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2 * delta)   # −∂V/∂y

E_mag = np.sqrt(Ex**2 + Ey**2)

# ─── Gráficas ────────────────────────────────────────────────────────────────
x = np.arange(N) * delta
y = np.arange(N) * delta
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(14, 10))

# Panel 1 — Potencial 3D
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X, Y, V, cmap='coolwarm', edgecolor='none', alpha=0.9)
ax1.set(xlabel='x [u.r.]', ylabel='y [u.r.]', zlabel='V [u.r.]',
        title='Potencial eléctrico V(x,y)')

# Panel 2 — Equipotenciales + densidad de carga
ax2 = fig.add_subplot(2, 2, 2)
cf = ax2.contourf(X, Y, V, levels=50, cmap='coolwarm')
ax2.contour(X, Y, V, levels=20, colors='black', linewidths=0.5, alpha=0.6)
# Mostrar las placas como rectángulos de color
for (fila, col, color, label) in [
        (slice(15,35), slice(15,35), 'red',   '+ρ₀'),
        (slice(15,35), slice(65,85), 'blue',  '−ρ₀'),
        (slice(65,85), slice(15,35), 'blue',  '−ρ₀'),
        (slice(65,85), slice(65,85), 'red',   '+ρ₀')]:
    xs = [col.start*delta, col.stop*delta, col.stop*delta, col.start*delta, col.start*delta]
    ys = [fila.start*delta, fila.start*delta, fila.stop*delta, fila.stop*delta, fila.start*delta]
    ax2.plot(xs, ys, color=color, lw=2)
plt.colorbar(cf, ax=ax2, label='V [u.r.]')
ax2.set(xlabel='x', ylabel='y', title='Líneas equipotenciales', aspect='equal')

# Panel 3 — Campo eléctrico (quiver diezmado)
ax3 = fig.add_subplot(2, 2, 3)
paso = 5   # diezmar para visualización
ax3.contour(X, Y, V, levels=20, colors='gray', linewidths=0.4, alpha=0.5)
ax3.quiver(X[::paso, ::paso], Y[::paso, ::paso],
           Ey[::paso, ::paso], Ex[::paso, ::paso],
           E_mag[::paso, ::paso], cmap='plasma', scale=500)
ax3.set(xlabel='x', ylabel='y', title='Campo eléctrico E = −∇V', aspect='equal')

# Panel 4 — Convergencia SOR
ax4 = fig.add_subplot(2, 2, 4)
ax4.semilogy(historial_error, 'steelblue', lw=2)
ax4.axhline(tol, color='red', linestyle='--', label=f'tolerancia = {tol}')
ax4.set(xlabel='Iteración', ylabel='Error máx. |ΔV|',
        title=f'Convergencia SOR  (ω = {omega})')
ax4.legend()
ax4.grid(True, alpha=0.3)

fig.suptitle('Cuadrupolo de placas cargadas en caja conductora aterrizada\n'
             'Ecuación de Poisson → SOR', fontsize=13)
plt.tight_layout()
plt.show()
