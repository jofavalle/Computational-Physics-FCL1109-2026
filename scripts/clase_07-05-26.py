# Script correspondiente a la clase del 07/05/2026, donde se introduce la ecuación de Navier-Stokes con vorticidad. Partimos y hacemos uso del archivo de la clase anterior clase050526.ipynb.
# Uso de los archivos clase050526.ipynb y clase070526.ipynb para la explicación de la ecuación de Navier-Stokes con vorticidad.

import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
Nx, Ny = 70, 24
h = 1.0
V0 = 1.0
omega = 0.3 # Cambiar a 0.3 para SOR
R = 0.1
tol = 1e-3
max_iter = 5000
L_beam = 8.0
H_beam = 4.0

u = np.zeros((Nx, Ny))
w = np.zeros((Nx, Ny))

# Posición de la viga
x0 = int(20)
x1 = x0 + int(L_beam / h)
y0 = Ny // 2 - int(H_beam / (2 * h))
y1 = y0 + int(H_beam / h)

def boundary_conditions():
    # Entrada 
    for j in range(Ny):
        u[0,j] = V0*j
    
    # Salida
    u[-1,:] = u[-2,:]
    w[-1,:] = w[-2,:]

    # Superficie superior
    u[:,-1] = V0*(Ny-1)

    # Inferior
    u[:,0] = 0

    # Viga
    u[x0:x1, y0:y1] = u[x0,y0]

def relax():
    max_residual = 0

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # Saltar interior de la viga
            if x0 <= i < x1 and y0 <= j < y1:
                continue

            u_new = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + w[i,j])
            ru = u_new - u[i,j]
            u[i,j] += omega * ru

            a1 = w[i+1,j] + w[i-1,j]
            a2 = w[i,j+1] + w[i,j-1]
            a3 = (R/4.0)*((u[i,j+1] - u[i,j-1]) * 
                          (w[i+1,j] - w[i-1,j]) - (u[i+1,j] - u[i-1,j]) * 
                          (w[i,j+1] - w[i,j-1]))
            w_new = 0.25 * (a1 + a2 + a3)
            rw = w_new - w[i,j]
            w[i,j] += omega * rw

            max_residual = max(max_residual, abs(ru), abs(rw))

    return max_residual

for it in range(max_iter):
    boundary_conditions()
    residual = relax()
    if it%100 == 0:
        print(f'Iteración {it}, Residual máximo: {residual:.5f}')
        print("u_upstream = ",
              u[x0-5, Ny//2])
        print("u arriba = ",
              u[x0 + int(L_beam) // 2, y1 + 2])
        print("u downstream = ",
              u[x1+5, Ny//2])
        print("Residual máximo: ", residual)
        print("---------------------------------------------------")

    if residual < tol:
        print(f'Convergencia alcanzada en {it} iteraciones.')
        break

# Visualización de los resultados
# función de corriente u(x,y)
# vorticidad w(x,y)
# curvas de nivel
# superficies 3D

vx = np.zeros_like(u)
vy = np.zeros_like(u)
vx[:,1:-1] = (u[:,2:] - u[:,:-2])/2
vy[1:-1,:] = (u[2:,:] - u[:-2,:])/2

X, Y = np.meshgrid(np.arange(Nx)*h, np.arange(Ny)*h)

plt.figure(figsize=(10, 5))
plt.contourf(
    X,Y,u.T,levels=40
)
plt.colorbar(label='u(x,y)')
plt.title('Función de corriente u(x,y)')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(figsize=(10, 5))
plt.contour(
    X,Y,w.T,levels=40
)
plt.colorbar(label='w(x,y)') #También lleva f
plt.title('Función de vorticidad w(x,y)')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(figsize=(10, 5))
plt.quiver(
    X,Y,vx.T,vy.T
)
plt.colorbar(label='w(x,y)')
plt.title('Función de vorticidad w(x,y)')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(figsize=(10, 5))
plt.streamplot(
    X,Y,vx.T,vy.T,density=1.5
)
plt.colorbar(label='w(x,y)')
plt.title('Función de vorticidad w(x,y)')
plt.xlabel('x')
plt.ylabel('y')
'''
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Campo de velocidad u')
plt.imshow(u.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto')
plt.colorbar(label='Velocidad u')
plt.subplot(1, 2, 2)
plt.title('Campo de vorticidad w')
plt.imshow(w.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto')
plt.colorbar(label='Vorticidad w')
plt.tight_layout()
'''
plt.show()

