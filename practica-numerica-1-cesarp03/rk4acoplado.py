#RK4 acoplado
import matplotlib.pyplot as plt
import numpy as np
def f(r,t):
	x = r[0]
	y = r[1]
	fx = x*y-x
	fy = y - x*y + np.sin(t)**2
	return np.array([fx, fy], float)

a = 0.0
b = 10.0
N = 1000
h = (b-a)/N

tpoints = np.arange(a, b, h)
xpoints=[]
ypoints=[]

r = np.array([1.0, 1.0], float)

for t in tpoints:
	xpoints.append(r[0])
	ypoints.append(r[1])

	k1 = h*f(r,t)
	k2 = h*f(r+0.5*k1, t + 0.5*h)
	k3 = h*f(r+0.5*k2, t+0.5*h)
	k4 = h*f(r+k3, t+h)
	
	r+=(k1+2*k2+2*k3+k4)/6

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

ax1.plot(tpoints,xpoints, color="red")
ax1.set_title("Solución x")
ax1.set_xlabel("t")
ax1.set_ylabel("x(t)")

ax2.plot(tpoints,ypoints, color="blue")
ax2.set_title("Solución y")
ax2.set_xlabel("t")
ax2.set_ylabel("y(t)")

plt.tight_layout()
plt.show()

