"""
SECCIÓN: VISUALIZACIÓN (matplotlib)
=====================================
Recetario completo de todos los patrones de graficación que aparecen
en los scripts y notebooks de la unidad 4.

ÍNDICE:
  V1.  Figura simple: plot con múltiples líneas (instantáneas de onda)
  V2.  Subplots (2×2 y 1×2)
  V3.  imshow — campo 2D como mapa de calor
  V4.  contourf + contour — equipotenciales y lineas de nivel
  V5.  streamplot — líneas de corriente (N-S)
  V6.  quiver — campo vectorial con flechas
  V7.  stem — espectro de amplitudes (modos / FFT)
  V8.  semilogy — escala logarítmica
  V9.  plot_surface 3D — superficie (Laplace, ring soliton)
  V10. FuncAnimation — animación de una línea
  V11. meshgrid — construcción estándar para 2D
  V12. colorbar — agregar barra de colores
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Se asumen variables de ejemplo para que el código sea ejecutable:
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x)
N = 50
V = np.random.rand(N, N)
u2d = np.random.rand(N, N)
X, Y = np.meshgrid(np.arange(N), np.arange(N))
vx = np.random.rand(N, N) - 0.5
vy = np.random.rand(N, N) - 0.5


# =============================================================================
# V1. PLOT CON MÚLTIPLES LÍNEAS — instantáneas de la onda en distintos tiempos
# =============================================================================
# Patrón: iterar sobre los tiempos de interés y graficar cada uno con label

fig, ax = plt.subplots(figsize=(10, 5))

for paso in [0, 50, 100, 200]:
    t_snap = paso * 0.005
    ax.plot(x, y * np.cos(paso * 0.05), label=f't = {t_snap:.3f} s')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Evolución temporal de la onda')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# V2. SUBPLOTS — cuatro paneles (2×2) o dos paneles (1×2)
# =============================================================================

# --- 2 filas × 2 columnas ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, y)
axes[0, 0].set(xlabel='x [m]', ylabel='u(x)', title='Panel A')

axes[0, 1].plot(x, y**2, color='orange')
axes[0, 1].set(xlabel='x [m]', ylabel='u²', title='Panel B')

axes[1, 0].plot(x, np.abs(y))
axes[1, 0].set(xlabel='x [m]', ylabel='|u|', title='Panel C')

axes[1, 1].plot(x, np.cumsum(y) * (x[1] - x[0]))
axes[1, 1].set(xlabel='x [m]', ylabel='∫u dx', title='Panel D')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)   # grilla en todos los paneles

fig.suptitle('Título global de la figura', fontsize=13)
plt.tight_layout()
plt.show()

# --- 1 fila × 2 columnas ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(x, y)
axes[1].plot(x, y**2)
plt.tight_layout()
plt.show()


# =============================================================================
# V3. IMSHOW — campo 2D como mapa de calor
# =============================================================================
# ⚠ imshow muestra la TRANSPUESTA visual: u[i,j] → columna i, fila j
#   Para que x sea horizontal e y vertical: usar u.T con origin='lower'

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    u2d.T,               # transponer para orientación correcta
    origin='lower',      # y=0 en la parte inferior (físico)
    extent=[0, 1, 0, 1], # [x_min, x_max, y_min, y_max]
    aspect='auto',       # ajustar relación de aspecto
    cmap='viridis',      # mapa de colores: 'viridis', 'plasma', 'RdBu_r', 'coolwarm'
    vmin=-1.0, vmax=1.0  # rango de colores fijo (útil para animaciones)
)
plt.colorbar(im, ax=ax, label='u(x,y)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Campo 2D — imshow')
plt.tight_layout()
plt.show()

# Variante sin extent (usa índices de pixel como ejes):
plt.imshow(u2d, cmap='viridis')
plt.colorbar()
plt.title('Campo 2D')
plt.show()


# =============================================================================
# V4. CONTOURF + CONTOUR — equipotenciales y líneas de nivel
# =============================================================================
# contourf: relleno de color por nivel
# contour:  líneas negras encima (más legible)

fig, ax = plt.subplots(figsize=(8, 6))

cf = ax.contourf(X, Y, V, levels=40, cmap='coolwarm')   # relleno
plt.colorbar(cf, ax=ax, label='V [V]')

ax.contour(X, Y, V, levels=20, colors='black', linewidths=0.5)   # líneas

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Líneas equipotenciales')
plt.tight_layout()
plt.show()

# Variante: solo contourf (sin líneas) — para N-S vorticidad
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, u2d, levels=40)
plt.colorbar(label='ω(x,y)')
plt.title('Vorticidad')
plt.xlabel('x'); plt.ylabel('y')
plt.show()


# =============================================================================
# V5. STREAMPLOT — líneas de corriente (N-S)
# =============================================================================
# ⚠ streamplot espera X, Y con meshgrid, y los campos vx, vy con la misma orientación

fig, ax = plt.subplots(figsize=(10, 5))
speed = np.sqrt(vx**2 + vy**2)   # magnitud para colorear

strm = ax.streamplot(
    X, Y,             # grilla (meshgrid)
    vx, vy,           # componentes del campo vectorial
    color=speed,      # colorear por velocidad (o usar color='blue')
    cmap='plasma',    # mapa de colores
    density=1.5,      # densidad de líneas (1.0 = normal)
    linewidth=1.0
)
plt.colorbar(strm.lines, ax=ax, label='|v| [m/s]')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Líneas de corriente')
plt.tight_layout()
plt.show()


# =============================================================================
# V6. QUIVER — campo vectorial con flechas
# =============================================================================
# Se suele submuestrear para no saturar la figura

paso_flecha = max(1, N // 15)   # mostrar 1 de cada n flechas

fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(
    X[::paso_flecha, ::paso_flecha],
    Y[::paso_flecha, ::paso_flecha],
    vx[::paso_flecha, ::paso_flecha],
    vy[::paso_flecha, ::paso_flecha],
    color='darkblue',
    scale=500         # ajustar para que las flechas no se superpongan
)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Campo de velocidades (quiver)')
plt.tight_layout()
plt.show()


# =============================================================================
# V7. STEM — espectro de amplitudes (modos normales / FFT)
# =============================================================================
n_modos = np.arange(1, 11)
amplitudes = np.exp(-0.3 * n_modos)   # ejemplo

fig, ax = plt.subplots(figsize=(9, 4))
ax.stem(n_modos, amplitudes,
        basefmt='k-',       # línea base negra
        markerfmt='C1o',    # marcadores color C1 (naranja)
        linefmt='C0-')      # líneas color C0 (azul)
ax.set_xlabel('Modo n')
ax.set_ylabel('|Bₙ|')
ax.set_title('Espectro de modos normales')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# V8. SEMILOGY — escala logarítmica en el eje y (errores, potencia espectral)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(x, np.abs(y) + 1e-14, color='red')   # +1e-14 evita log(0)
ax.set_xlabel('x [m]')
ax.set_ylabel('Error (escala log)')
ax.set_title('Error vs posición')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# V9. PLOT_SURFACE 3D — superficie (Laplace, ring soliton, clase_28-04-26)
# =============================================================================
fig = plt.figure(figsize=(10, 7))
ax3d = fig.add_subplot(111, projection='3d')

ax3d.plot_surface(
    X, Y, V,
    cmap='viridis',
    alpha=0.9       # transparencia (0=invisible, 1=sólido)
)
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('V')
ax3d.set_title('Potencial eléctrico — Superficie 3D')
plt.show()

# Variante: 4 superficies en subplots 2×2 (clase_28-04-26)
fig = plt.figure(figsize=(12, 8))
for idx in range(4):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')   # índice 1-based
    ax.plot_surface(X, Y, V * (idx + 1) * 0.25, cmap='viridis')
    ax.set_title(f'Paso {idx * 5}')
plt.show()


# =============================================================================
# V10. FUNCANIMATION — animar una línea 1D (onda, Burgers)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(x, y)          # coma después de line: desempaqueta la tupla
ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x [m]')
ax.set_ylabel('u(x,t)')
ax.set_title('Animación de la onda')

def update(frame):
    global u_actual, u_anterior, u_nuevo, r2
    # --- avanzar el campo un paso ---
    u_nuevo[1:-1] = (
        2 * u_actual[1:-1]
        - u_anterior[1:-1]
        + r2 * (u_actual[2:] - 2 * u_actual[1:-1] + u_actual[:-2])
    )
    u_nuevo[0] = 0.0; u_nuevo[-1] = 0.0
    u_anterior[:] = u_actual
    u_actual[:]   = u_nuevo
    # --- actualizar la línea ---
    line.set_ydata(u_actual)
    ax.set_title(f'frame = {frame}')
    return line,               # debe devolver iterable de artistas

ani = FuncAnimation(
    fig,
    update,
    frames=300,     # número de frames
    interval=20,    # ms entre frames
    blit=True       # solo redibujar lo que cambia (más rápido)
)
plt.show()

# Variante: avanzar VARIOS pasos por frame (más rápida la simulación)
def update_rapido(frame):
    for _ in range(5):   # 5 pasos físicos por frame
        # ... avanzar campo ...
        pass
    line.set_ydata(u_actual)
    return line,


# =============================================================================
# V10b. FUNCANIMATION — animar un campo 2D con imshow
# =============================================================================
fig2, ax2 = plt.subplots()
im_anim = ax2.imshow(u2d, animated=True, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im_anim)

def update_2d(frame):
    # ... avanzar campo 2D ...
    im_anim.set_data(u2d)    # actualizar datos del imshow
    return [im_anim]

ani2 = FuncAnimation(fig2, update_2d, frames=200, interval=30)
plt.show()


# =============================================================================
# V11. MESHGRID — construcción estándar para campos 2D
# =============================================================================
Nx, Ny = 50, 40
dx, dy = 0.4, 0.4

x_1d = np.arange(Nx) * dx          # [0, dx, 2dx, ..., (Nx-1)*dx]
y_1d = np.arange(Ny) * dy

# indexing='ij': X[i,j]=x_i, Y[i,j]=y_j  (natural para matrices físicas)
X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')   # forma (Nx, Ny)

# indexing='xy' (por defecto): X[i,j]=x_j, Y[i,j]=y_i  (orientación gráfica)
X_xy, Y_xy = np.meshgrid(x_1d, y_1d)             # forma (Ny, Nx)

# ⚠ Al pasar a imshow/contourf/streamplot, verificar orientación con u.T si hace falta


# =============================================================================
# V12. COLORBAR — agregar barra de colores (resumen de usos)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Caso 1: a partir del objeto retornado por imshow
im1 = axes[0].imshow(V, cmap='viridis')
plt.colorbar(im1, ax=axes[0], label='V [V]')

# Caso 2: a partir de contourf
cf2 = axes[1].contourf(X_xy, Y_xy, V.T, levels=30, cmap='coolwarm')
plt.colorbar(cf2, ax=axes[1], label='V [V]')

# Caso 3: a partir de streamplot
strm3 = axes[2].streamplot(X_xy, Y_xy, vx.T, vy.T, color=np.sqrt(vx.T**2+vy.T**2))
plt.colorbar(strm3.lines, ax=axes[2], label='|v|')

plt.tight_layout()
plt.show()


# =============================================================================
# TRUCOS RÁPIDOS
# =============================================================================
# ax.set(xlabel='x', ylabel='y', title='T')  → equivale a 3 líneas separadas
# for ax in axes.flat: ax.grid(...)          → aplicar a todos los paneles
# plt.tight_layout()                         → evitar solapamiento de etiquetas
# fig.suptitle('Título', fontsize=13)        → título global sobre subplots
# ax.set_xlim(a, b); ax.set_ylim(c, d)      → limitar rango de ejes
# ax.axvline(x0, color='r', ls='--')        → línea vertical de referencia
# ax.axhline(y0, color='r', ls='--')        → línea horizontal de referencia
# ax.add_patch(plt.Rectangle((x0,y0),w,h, color='gray')) → dibujar rectángulo (viga)
