import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
L = 1.0
c = 1.0

N = 751
Nt = 751
dt = 0.001 # para r < 1
# dt = 0.01 # para r > 1
dx = L / (N - 1)
# si r = 1
#dt = dx
r = (c*dt)/(dx)

print(r)

# Condiciones iniciales
t = np.arange(N)*dt

x = np.linspace(0,L,N)

#y1 = np.sin((np.pi / L)*x)
#y2 = 0.5*np.sin((3*np.pi / L)*x)
#plt.plot(x,y1)
#plt.plot(x,y2)
#plt.show()

y = np.sin((np.pi / L)*x) + 0.5*np.sin((3*np.pi / L)*x)

# Agregando nuevo modo
#y = np.sin((np.pi / L)*x) + 0.5*np.sin((3*np.pi / L)*x) + 3*np.sin((5*np.pi / L)*x)

# Condiciones de frontera
y[0] = 0.0
y[-1] = 0.0

# Pasos a guardar
#pasos_t = [0,0.25,0.50,0.75]
snaps = {0: y.copy()}

plt.plot(x,snaps[0])

# Leapfrog
y_old = y.copy()
y_new = np.zeros(N)
for t in range(Nt):
	for i in range(1,N-1):
		y_new[i] = 2*y[i] - y_old[i] + (r**2)*(y[i+1] + y[i-1] - 2*y[i])

	#CC
	y_new[0] = 0.0
	y_new[-1] = 0.0

	if((t*dt)==0.25 or (t*dt)==0.50 or (t*dt)==0.75):
		snaps[t*dt] = y_new.copy()
		print(t*dt)

	y_old = y.copy()
	y = y_new.copy()

plt.plot(x,snaps[0.25])
plt.plot(x,snaps[0.50])
plt.plot(x,snaps[0.75])
plt.show()
