# César Elías Peña Aparicio

# PROBLEMA 4

import matplotlib.pyplot as plt
import numpy as np

# Definimos constantes
k = 0.8
g = 9.8
n_array = np.array([1, 3/2, 2])

# Definimos la función
def f(r, t, n):
	x = r[0]
	y = r[1]
	vx = r[2]
	vy = r[3]
	v_mag = np.sqrt(vx**2 + vy**2)
	dx = vx
	dy = vy
	dvx = -k*(v_mag**(n-1))*vx
	dvy = -k*(v_mag**(n-1))*vy - g
	return np.array([dx, dy, dvx, dvy], float)

# Definimos los parametros del método
a = 0.0
b = 25.0
N = 10000
h = (b-a)/N

# Definimos el intervalo temporal
tpoints = np.arange(a, b, h)
# Implementamos el método RK4 para cada n

# Preparamos una sola grafíca para las 3 trayectorias
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for n in n_array:
    # Definimos las condiciones iniciales
    r = np.array([0.0, 0.0, 18.23, 12.3], float)
    # Definimos las listas
    xpoints=[]
    ypoints=[]
    vxpoints=[]
    vypoints=[]

    for t in tpoints:
        if r[1] < 0 and t > 0: #Si y < 0 entonces el bucle se rompe, es decir, el proyectil llegó al suelo
             break
        
        xpoints.append(r[0])
        ypoints.append(r[1])
        vxpoints.append(r[2])
        vypoints.append(r[3])

        k1 = h*f(r, t, n)
        k2 = h*f(r+0.5*k1, t + 0.5*h, n)
        k3 = h*f(r+0.5*k2, t+0.5*h, n)
        k4 = h*f(r+k3, t+h, n)
        
        r+=(k1+2*k2+2*k3+k4)/6
        
    vx = np.array(vxpoints)
    vy = np.array(vypoints)
    vmag = np.sqrt(vx**2 + vy**2)
    # Ploteamos esta trayectoria antes de pasar al siguiente n
    ax1.plot(xpoints, ypoints, label=f"n = {n}")
    ax2.plot(tpoints[:len(vmag)], vmag, label=f"n = {n}")


# Ploteamos el caso ideal sin fricción:
# Condiciones iniciales
x0 = 0.0
y0 = 0.0
v0x = 18.23
v0y = 12.3

t_vuelo_ideal = (2*v0y)/g
t_ideal = np.linspace(0, t_vuelo_ideal, 1000)

xideal = v0x * t_ideal
yideal = v0y * t_ideal-((1/2) * g * (t_ideal**2))



# Customización del plot de posición
ax1.plot(xideal, yideal, label="Ideal (k = 0)", linestyle="--", color="black", linewidth=2)
ax1.set_title("Trayectorias de Tiro Parabólico con Fricción")
ax1.set_xlabel("Distancia horizontal x (m)")
ax1.set_ylabel("Altura y (m)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
# Fijar límite inferior de y a 0 para que sea "el suelo"
ax1.set_ylim(bottom=0)

# Customización del plot de velocidades
ax2.set_title("Velocidades de Tiro Parabólico con Fricción")
ax2.set_xlabel("Tiempo (t)")
ax2.set_ylabel("Magnitud de la velocidad (m/s)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
# Fijar límite inferior de y a 0 para que sea "el suelo"
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("Posición y Velocidad")
plt.show()

