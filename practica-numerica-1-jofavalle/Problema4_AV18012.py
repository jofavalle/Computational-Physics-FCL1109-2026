# Practica Numerica 1 - Problema 4
# Fisica Computacional, Ciclo I 2026
# Carnet: AV18012
# Tiro parabolico con resistencia del aire, Runge-Kutta de orden 4.
# Como la friccion es proporcional a m, la masa se cancela:
#   a = -k |v|^(n-1) v - g ey

import numpy as np
import matplotlib.pyplot as plt

g = 9.81
k = 0.8
v0x, v0y = 18.23, 12.3
casos = [1.0, 1.5, 2.0]


# estado s = [x, y, vx, vy]
def deriv(t, s, n):
    x, y, vx, vy = s
    v = np.hypot(vx, vy)
    fac = k * v**(n-1)
    return np.array([vx, vy, -fac*vx, -fac*vy - g])


def rk4(f, s0, t0, tf, dt, n):
    pasos = int(round((tf-t0)/dt))
    t = np.linspace(t0, t0+pasos*dt, pasos+1)
    s = np.empty((pasos+1, len(s0)))
    s[0] = s0
    for i in range(pasos):
        k1 = f(t[i], s[i], n)
        k2 = f(t[i] + dt/2, s[i] + dt/2*k1, n)
        k3 = f(t[i] + dt/2, s[i] + dt/2*k2, n)
        k4 = f(t[i] + dt, s[i] + dt*k3, n)
        s[i+1] = s[i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, s


# corto la trayectoria cuando y se vuelve negativo (toca el suelo)
def hasta_el_suelo(t, x, y):
    bajo = np.where(y[1:] < 0)[0]
    if len(bajo) == 0:
        return t, x, y
    j = bajo[0] + 1
    frac = y[j-1] / (y[j-1] - y[j])      # interpolo el punto de impacto
    xi = x[j-1] + frac*(x[j]-x[j-1])
    ti = t[j-1] + frac*(t[j]-t[j-1])
    return np.append(t[:j], ti), np.append(x[:j], xi), np.append(y[:j], 0.0)


s0 = np.array([0.0, 0.0, v0x, v0y])
sol = {n: rk4(deriv, s0, 0.0, 25.0, 0.005, n) for n in casos}
color = {1.0: 'tab:blue', 1.5: 'tab:green', 2.0: 'tab:red'}

# (b) trayectorias y vs x, mas el caso sin friccion (Fisica I)
plt.figure(figsize=(10, 6))
for n in casos:
    t, s = sol[n]
    tr, xr, yr = hasta_el_suelo(t, s[:, 0], s[:, 1])
    plt.plot(xr, yr, color=color[n], lw=1.8, label="n = %g" % n)

tsuelo = 2*v0y/g
ta = np.linspace(0, tsuelo, 300)
plt.plot(v0x*ta, v0y*ta - 0.5*g*ta**2, 'k--', lw=2, label='sin friccion')
plt.axhline(0, color='gray', lw=0.6)
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.title('Trayectorias del proyectil (RK4)')
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("P4_trayectorias.png", dpi=130)
plt.close()

# (c) |v| vs t. La linea punteada es la velocidad terminal teorica
plt.figure(figsize=(10, 6))
for n in casos:
    t, s = sol[n]
    vmod = np.hypot(s[:, 2], s[:, 3])
    plt.plot(t, vmod, color=color[n], lw=1.6, label="n = %g" % n)
    vt = (g/k)**(1.0/n)        # de k|v|^n = g
    plt.axhline(vt, color=color[n], ls=':', lw=1.0)
plt.xlabel('t [s]'); plt.ylabel('|v| [m/s]')
plt.title('Rapidez en funcion del tiempo')
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("P4_velocidades.png", dpi=130)
plt.close()

# resumen rapido
for n in casos:
    t, s = sol[n]
    tr, xr, yr = hasta_el_suelo(t, s[:, 0], s[:, 1])
    print("n =", n, " vel terminal =", round((g/k)**(1/n), 3),
          " alcance =", round(xr[-1], 3), "m")
print("sin friccion: alcance =", round(v0x*tsuelo, 3),
      "m, altura max =", round(v0y**2/(2*g), 3), "m")
