
import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
Nx, Ny = 60, 60
h = 0.4
V0 = 8e-4
g = 9.8
omega = 0.1 
nu = 0.5
R = V0 * h / nu
Niter = 4000
Nb = 15
Ndown = 20

u = np.zeros((Nx+1, Ny+1))
w = np.zeros((Nx+1, Ny+1))

def BelowHole():
    for i in range(Nb+1,Nx+1):
        u[i,0] = u[i-1,1]
        w[i-1,0] = w[i-1,1]
        for j in range(0,Ndown+1):
            if i == Nb:
                vy=0
            elif i == Nx:
                vy = -np.sqrt(2*g*h*(Ny+Nb-j))
            elif i == Nx-1:
                vy = -np.sqrt(2*g*h*(Ny+Nb-j))/2
            else: 
                vy = 0
            u[i,j] = u[i-1,j] - vy*h

def BorderRight():
    for j in range(1,Ny+1):
        vy = -np.sqrt(2*g*h*(Ny-1))
        u[Nx,j] = u[Nx-1,j] + vy*h
        u[Nx,j] = u[Nx,j-1]
        w[Nx,j] = -2*(u[Nx,j] - u[Nx,j-1])/h**2

def BottomBefore():
    for i in range(1,Nb+1):
        u[i,Ndown] = u[i,Ndown-1]
        w[i,Ndown] = -2*(u[i,0]-u[i,1])

def Top():
    for i in range(1,Nx):
        u[i,Ny] = u[i,Ny-1]
        w[i,Ny] = 0

def Left():
    for j in range(Ndown,Ny):
        w[0,j] = -2*(u[0,j]-u[1,j])/h**2
        u[0,j] = u[1,j]

def Borders():
    BelowHole()
    BorderRight()
    BottomBefore()
    Top()
    Left()

def relax():
    Borders()
    for i in range(1,Nx):
        for j in range(1,Ny):
            r1 = omega*(
                (
                    u[i+1,j]
                    + u[i-1,j]
                    + u[i,j+1]
                    + u[i,j-1]
                    - h**2*w[i,j]
                )*0.25 - u[i,j]
            )
            u[i,j] += r1
    Borders()
    for i in range(1,Nx):
        for j in range(1,Ny):
            a1 = ( 
                w[i+1,j]
                + w[i-1,j]
                + w[i,j+1]
                + w[i,j-1]
            )
            a2 = (
                (u[i,j+1] - u[i,j-1])*(w[i+1,j]-w[i-1,j])
            )
            a3 = (
                (u[i+1,j] - u[i-1,j])*(w[i,j+1]-w[i,j-1])
            )
            r2 = omega*((
                a1 + (R/4.0)*(a3-a2)
                ) / 4.0
                -w[i,j])
            w[i,j] += r2

# Iteración principal
for it in range(Niter):
    relax()
    if it % 100 == 0:
        print(f"Iteración {it}/{Niter}")
    
# Velocidades
vx = np.zeros_like(u)
vy = np.zeros_like(u)

for i in range(1,Nx):
    for j in range(1,Ny):
        vx[i,j] = (u[i,j+1] - u[i,j-1])/(2*h)
        vy[i,j] = -((u[i+1,j] - u[i-1,j]))/(2*h)

x = np.arange(0,Nx+1)*h
y = np.arange(0,Ny+1)*h

X,Y = np.meshgrid(x,y)

plt.figure(figsize=(8,6))
plt.imshow(
    u.T,
    origin='lower',
    extent=[0,Nx*h,0,Ny*h],
    aspect='auto'
)
plt.colorbar(label='Velocidad u')
plt.title('Función de corriente')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(
    w.T,
    origin='lower',
    extent=[0,Nx*h,0,Ny*h],
    aspect='auto'
)
plt.colorbar(label='Vorticidad')
plt.title('Vorticidad')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8,6))
plt.streamplot(
    X, Y, vx.T, vy.T,
    density=1.5,
    color=np.sqrt(vx.T**2 + vy.T**2),
    cmap='viridis'
)
plt.colorbar(label='Velocidad')
plt.title('Campo de velocidades')
plt.xlabel('x')
plt.ylabel('y')
plt.show()