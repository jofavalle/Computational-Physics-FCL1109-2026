import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametros físicos
L = 50.0
alpha = 1.27
ti = 100.0
Tv = 10000.0

# Grid
N = 100
dx = L/(N-1)

dt = 0.01

r_target = (alpha * dt) / dx**2

assert r_target <= 0.25, f"¡Inestable en 2D! r = {r_target:.4f}"

Nt = 100000

T = np.zeros((N,N))

# Condiciones inciales

T[0,:] = ti # Fija la pared izquierda
T[-1,:] = ti # Fija la pared derecha
T[:, 0] = ti # Fija la pared arriba
T[:, -1] = ti # Fija la pared abajo

T[48:52, 48:52] = Tv

T_save = {0: T.copy()} 

# Algoritmo
for n in range(Nt):
    T_new = T.copy()
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + r_target * (T[2:,1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] -4*T[1:-1, 1:-1]) # No se entiende

    #T_new[0,:] = ti # Fija la pared izquierda
    #T_new[-1,:] = ti # Fija la pared derecha
    #T_new[:, 0] = ti # Fija la pared arriba
    #T_new[:, -1] = ti # Fija la pared abajo

    T_new[48:52, 48:52] = Tv

    T = T_new

    T_save[n + 1] = T.copy()

# Ordenar los pasos guardados y submuestrear para la animación
pasos = sorted(T_save.keys())
paso_anim = max(1, len(pasos) // 200)   # ~200 frames en total
frames = pasos[::paso_anim]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    T_save[frames[0]].T,
    origin='lower',
    extent=[0, L, 0, L],
    cmap='viridis',
    vmin=0.0,
    vmax=Tv
)
plt.colorbar(im, ax=ax, label='T [°C]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
titulo = ax.set_title('')

def actualizar(paso):
    im.set_data(T_save[paso].T)
    titulo.set_text(f'Difusión 2D FTCS  -  t = {paso * dt:.2f} s')
    return im, titulo

ani = FuncAnimation(
    fig,
    actualizar,
    frames=frames,
    interval=40,     # [ms] entre frames
    blit=True
)

plt.tight_layout()
plt.show()