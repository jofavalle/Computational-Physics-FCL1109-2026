import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def direccion_propagacion_desde_fases(phi_e, phi_b):
    """Determina la dirección media de propagación usando S_z ~ <E_x B_y>."""
    desfase = (phi_e - phi_b) % (2 * np.pi)
    flujo_medio = np.cos(desfase)

    if np.isclose(flujo_medio, 0.0, atol=1e-10):
        return "sin propagacion neta en z", desfase
    if flujo_medio > 0:
        return "+z", desfase
    return "-z", desfase


# Parámetros físicos
c = 1.0  # Velocidad de la luz
Nz = 300  # Número de puntos en la dirección z
dz = 1.0  # Paso espacial
beta = 0.4 # Condición de Courant para la estabilidad numérica
dt = beta * dz / c  # Paso de tiempo

steps = 600  # Número de pasos de tiempo

# Parámetros de la modulación senoidal y de fase
longitud_onda = 30.0
k = 2 * np.pi / longitud_onda
phi_E = 0.0
phi_B = 0.0

# Perfil del medio: vacio - dielectrico - vacio
eps_r = np.ones(Nz)
inicio_dielectrico = 120
fin_dielectrico = 180
eps_r_dielectrico = 4.0
eps_r[inicio_dielectrico:fin_dielectrico] = eps_r_dielectrico

# Campos
E_x = np.zeros(Nz)  # Campo eléctrico
B_y = np.zeros(Nz)  # Campo magnético

# Pulso inicial: envolvente gaussiana modulada por una portadora sinusoidal
z = np.arange(Nz)
envolvente = np.exp(-((z - 80) / 15)**2)
E_x = envolvente * np.sin(k * z + phi_E)
B_y = (envolvente * np.sin(k * z + phi_B)) / c

direccion, desfase = direccion_propagacion_desde_fases(phi_E, phi_B)

# Figura
fig, ax = plt.subplots(figsize=(10, 5))
lineE, = ax.plot(z, E_x, label='E_x')
lineB, = ax.plot(z, B_y, label='B_y')
ax.set_xlim(0, Nz)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('z')
ax.set_ylabel('Campo')
ax.legend()
ax.set_title('Pulso gaussiano modulado por una sinusoidal')
ax.axvspan(
    inicio_dielectrico,
    fin_dielectrico,
    color='grey',
    alpha=0.75,
    label='Dielectrico'
)
ax.text(
    0.02,
    0.95,
    f'phi_E = {phi_E:.2f} rad\nphi_B = {phi_B:.2f} rad\n'
    f'desfase = {desfase:.2f} rad\nDireccion = {direccion}\n'
    f'epsilon_r = {eps_r_dielectrico:.1f} en {inicio_dielectrico} <= z < {fin_dielectrico}',
    transform=ax.transAxes,
    va='top',
    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85}
)
ax.legend()

# Actualización FDTD
def update(frame):
    global E_x, B_y
    
    # En el dieléctrico el campo eléctrico evoluciona más lento por la mayor permitividad.
    E_x[1:-1] += (beta / eps_r[1:-1]) * (B_y[:-2] - B_y[1:-1])
    
    # Actualizar campo magnético
    B_y[1:-1] += -beta*(E_x[2:] - E_x[1:-1])

    #Condiciones de frontera
    E_x[0] = E_x[-1] = 0
    B_y[0] = B_y[-1] = 0
    
    lineE.set_ydata(E_x)
    lineB.set_ydata(B_y)
    return lineE, lineB

ani = FuncAnimation(
    fig, update, frames=steps, interval=20, blit=True
)

plt.show()