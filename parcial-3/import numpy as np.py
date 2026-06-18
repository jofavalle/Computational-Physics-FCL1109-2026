import numpy as np
import matplotlib.pyplot as plt

# Parametros
N = 100
delta = 1
epsilon0 = 1
rho0 = 100.0
omega = 1.85
tol = 1e-5
max_iterations = 5000
historial = []

x_centro, y_centro = 50, 50
radio = 35
voltaje_electrodo = 500

V = np.zeros((N,N))
rho = np.zeros((N,N))

# Condiciones iniciales

rho[15:35, 15:35] = rho0
rho[15:35, 65:85] = -rho0
rho[65:85, 15:35] = - rho0
rho[65:85, 65:85] = rho0

# Añadimos un electrodo:

def es_electrodo(i, j):
    """
    Función que nos dice si un punto (i, j) pertenece al electrodo.
    Es una forma muy humana de separar la lógica geométrica del algoritmo.
    """
    distancia_al_centro = np.sqrt((i - x_centro)**2 + (j - y_centro)**2)
    return distancia_al_centro <= radio

for iterations in range(max_iterations):

    V_old = V.copy()

    for i in range(1, N-1):
        for j in range(1, N-1):
            
            if es_electrodo(i,j):
                V[i, j] = voltaje_electrodo
                continue

            V_gs = 0.25*(V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1] + (rho[i, j] * delta**2)/epsilon0)
            V[i, j] = (1-omega) * V[i,j] + omega*V_gs

    error = np.max(np.abs(V-V_old))
    historial.append(error)
    if iterations % 100 == 0:
        print(iterations, " ", error)
    if error < tol:
        print(f"Convergencia alcanzada en la iteracion {iterations}")
        break

Ex = np.zeros_like(V)
Ey = np.zeros_like(V)
for i in range(1,N-1):
    for j in range(1,N-1):
        Ex[i, j] = - (V[i+1, j] - V[i-1, j])/(2*delta)
        Ey[i, j] = - (V[i, j+1] - V[i, j-1])/(2*delta)

Emag = np.sqrt(Ex**2 + Ey**2)


x = np.arange(N) * 1
y = np.arange(N) * 1

X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X, Y, V, cmap ='viridis')
ax2 = fig.add_subplot(2, 2, 2)
ax2.semilogy(historial)
ax2.axhline(tol, color='red', alpha = 0.7)
ax2.grid(True)
ax3 = fig.add_subplot(2, 2, 3)
ax3.contour(X, Y, V, colors='gray')
ax3.quiver(X, Y, Ex, Ey, Emag, cmap='plasma')
ax4 = fig.add_subplot(2, 2, 4)
ax4.contourf(X, Y, V, cmap ='viridis')
ax4.contour(X, Y, V, colors='black')


plt.tight_layout()
plt.show()