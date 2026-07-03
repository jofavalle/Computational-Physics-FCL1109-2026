"""
PLANTILLA 06b - Ecuación de calor - Tres estilos de visualización
==================================================================
Misma simulación FTCS que 06_calor_ftcs.py, pero con tres figuras:

  Figura 1 - Snapshots + error (visualización original de la plantilla)
  Figura 2 - Mapa espacio-tiempo con imshow
  Figura 3 - Animación con FuncAnimation

Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L     = 1.0    # [m]     longitud de la barra
alpha = 0.01   # [m²/s]  difusividad térmica
T_izq = 100.0  # [°C]    temperatura en x=0 (Dirichlet)
T_der =   0.0  # [°C]    temperatura en x=L (Dirichlet)
T_ini =   0.0  # [°C]    temperatura inicial uniforme

# =============================================================================
# GRILLA ESPACIAL Y TEMPORAL
# =============================================================================
Nx = 100
dx = L / (Nx - 1)
r_target = 0.4
dt       = r_target * dx**2 / alpha
Nt       = 3000

r = alpha * dt / dx**2
assert r <= 0.5, f"¡Inestable! r = {r:.4f} > 0.5"
print(f"r = {r:.4f}  (Von Neumann estable)")
print(f"dt = {dt:.6f} s,  tiempo total = {Nt*dt:.4f} s")

x = np.linspace(0, L, Nx)

# =============================================================================
# CONDICIÓN INICIAL
# =============================================================================
T = np.full(Nx, T_ini)
T[0]  = T_izq
T[-1] = T_der

# =============================================================================
# ALGORITMO CENTRAL - FTCS
# Guardamos TODA la evolución en T_matrix para el mapa y la animación.
# Los snapshots para la figura 1 se extraen de T_matrix al final.
# =============================================================================
T_matrix = np.zeros((Nt, Nx))   # T_matrix[n, i] = temperatura en paso n, nodo i
T_matrix[0] = T.copy()

for n in range(1, Nt):
    T_nuevo = T.copy()
    T_nuevo[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
    T_nuevo[0]  = T_izq
    T_nuevo[-1] = T_der
    T = T_nuevo
    T_matrix[n] = T.copy()

# =============================================================================
# SOLUCIÓN ESTACIONARIA ANALÍTICA
# =============================================================================
T_estacionaria = T_izq + (T_der - T_izq) * x / L

# =============================================================================
# FIGURA 1 - Snapshots + panel de error (visualización original)
# =============================================================================
pasos_snap = [0, Nt // 10, Nt // 4, Nt // 2, Nt - 1]

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes1[0]
for paso in pasos_snap:
    t_snap = paso * dt
    ax.plot(x, T_matrix[paso], label=f't = {t_snap:.3f} s')
ax.plot(x, T_estacionaria, 'k--', linewidth=2, label='Estacionaria (analítica)')
ax.set_xlabel('x [m]')
ax.set_ylabel('T [°C]')
ax.set_title(f'Difusión FTCS  (α={alpha}, r={r:.2f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes1[1]
for paso in pasos_snap:
    if paso == 0:
        continue
    t_snap = paso * dt
    error  = np.abs(T_matrix[paso] - T_estacionaria)
    ax2.semilogy(x, error + 1e-14, label=f't = {t_snap:.3f} s')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('|T - T_est| [°C]')
ax2.set_title('Error respecto a solución estacionaria')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig1.suptitle('Figura 1 - Snapshots temporales', fontsize=13)
plt.tight_layout()

# =============================================================================
# FIGURA 2 - Mapa espacio-tiempo con imshow
# Eje X = posición a lo largo de la barra
# Eje Y = tiempo (de abajo = t=0 hacia arriba = t final)
# Color = temperatura
# =============================================================================
fig2, ax3 = plt.subplots(figsize=(9, 6))

im = ax3.imshow(
    T_matrix,
    origin='lower',          # t=0 abajo, t_final arriba
    aspect='auto',           # escala ejes independientemente
    extent=[0, L, 0, Nt*dt], # [x_min, x_max, t_min, t_max]
    cmap='hot'
)
plt.colorbar(im, ax=ax3, label='T [°C]')
ax3.set_xlabel('x [m]')
ax3.set_ylabel('t [s]')
ax3.set_title(f'Mapa espacio-tiempo  (α={alpha}, r={r:.2f})')
ax3.grid(False)

# Línea horizontal en el último snapshot para referencia
ax3.axhline(y=(Nt - 1) * dt, color='cyan', linewidth=1, linestyle='--',
            label='t final')
ax3.legend(fontsize=9)

fig2.suptitle('Figura 2 - Mapa espacio-tiempo', fontsize=13)
plt.tight_layout()

# =============================================================================
# FIGURA 3 - Animación con FuncAnimation
# Se recorren frames cada 'paso_anim' pasos para que no sea demasiado lenta.
# =============================================================================
paso_anim = Nt // 150        # ~150 frames en total
frames    = range(0, Nt, paso_anim)

fig3, ax4 = plt.subplots(figsize=(8, 5))
linea,     = ax4.plot(x, T_matrix[0], color='orangered', linewidth=2)
linea_est, = ax4.plot(x, T_estacionaria, 'k--', linewidth=1.5,
                      label='Estacionaria (analítica)')

ax4.set_xlim(0, L)
ax4.set_ylim(T_der - 5, T_izq + 5)
ax4.set_xlabel('x [m]')
ax4.set_ylabel('T [°C]')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
titulo = ax4.set_title('')

def actualizar(n):
    """Actualiza la curva y el título en cada frame."""
    linea.set_ydata(T_matrix[n])
    titulo.set_text(f'Difusión FTCS  -  t = {n*dt:.4f} s')
    return linea, titulo

ani = FuncAnimation(
    fig3,
    actualizar,
    frames=frames,
    interval=30,     # [ms] entre frames
    blit=True
)

fig3.suptitle('Figura 3 - Animación', fontsize=13)
plt.tight_layout()

# Mostrar las tres figuras
plt.show()
