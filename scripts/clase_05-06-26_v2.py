import numpy as np
import matplotlib.pyplot as plt

alpha = 1/137.036
Z = 56.0
m = 139.57 # MeV/c^2

def v(r):
    rmin = 1e-6
    return -Z*alpha/max(r, rmin)

def rhs(r,y,E,l):
    u = y[0]
    up = y[1]

    coeff = (
        (E - v(r))**2 - m**2 - l*(l+1)/r**2
    )

    return np.array([up, -coeff * u])

def rk4_step(r, h, y, E, l):
    k1 = rhs(r, y, E, l)
    k2 = rhs(r + h/2, y + h/2 * k1, E, l)
    k3 = rhs(r + h/2, y + h/2 * k2, E, l)
    k4 = rhs(r + h, y + h * k3, E, l)

    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def mismatch(E,l):
    rmin = 1e-4
    rmax = 50.0

    N = 6000

    h = (rmax - rmin) / (N-1)

    rmatch = 15.0

    imatch = int((rmatch - rmin) / h)

    if l==0:
        yL = np.array([rmin, 1.0])
    else:
        yL = np.array([rmin**(l+1), (l+1)*rmin**l])

    for i in range(imatch):
        r = rmin + i*h
        yL = rk4_step(r, h, yL, E, l)
        scale = max(abs(yL[0]), abs(yL[1]),1.0)

        if scale > 1e40:
            yL /= scale

    # Guardar contra división por cero en la derivada logarítmica izquierda
    if abs(yL[0]) < 1e-30 or not np.isfinite(yL[0]):
        return np.inf
    left = yL[1] / yL[0]
    u_match = yL[0]

    kappa = np.sqrt(max(m**2 - E**2, 1e-10))

    yR = np.array([np.exp(-kappa*rmax), -kappa*np.exp(-kappa*rmax)])

    for i in range(N-1, imatch, -1):
        r = rmin + i*h
        yR = rk4_step(r, -h, yR, E, l)
        scale = max(abs(yR[0]), abs(yR[1]),1.0)

        if scale > 1e40:
            yR /= scale

    # Guardar contra división por cero en la derivada logarítmica derecha
    if abs(yR[0]) < 1e-30 or not np.isfinite(yR[0]):
        return np.inf
    right = yR[1] / yR[0]

    return left-right

def find_state(E1,E2,l,tol=1e-8):
    f1 = mismatch(E1,l)
    f2 = mismatch(E2,l)

    if f1 * f2 > 0:
        return None

    while abs(E2 - E1) > tol:
        Em = 0.5 * (E1 + E2)
        fm = mismatch(Em, l)
        if f1 * fm < 0:
            E2 = Em
            f2 = fm
        else:
            E1 = Em
            f1 = fm

    return 0.5 * (E1 + E2)

def search_states(l):
    Emin = 0.90*m
    Emax = 0.99999*m

    grid = np.linspace(Emin, Emax, 500)

    vals = []

    for E in grid:
        vals.append(mismatch(E, l))
    
    vals = np.array(vals)

    roots = []

    for i in range(len(vals)-1):
        if vals[i]*vals[i+1] < 0:
            root = find_state(grid[i], grid[i+1], l)
            if root is not None:
                roots.append(root)

    return roots

def wavefunction(E,l):
    rmin = 1e-4
    rmax = 50.0

    N = 8000

    h = (rmax - rmin) / (N-1)

    r = np.linspace(rmin, rmax, N)

    if l==0:
        y = np.array([rmin, 1.0])
    else:
        y = np.array([rmin**(l+1), (l+1)*rmin**l])

    u = np.zeros(N)

    u[0] = y[0]

    for i in range(N - 1):
        y = rk4_step(r[i], h, y, E, l)
        scale = max(abs(y[0]), abs(y[1]),1.0)

        if scale > 1e40:
            y /= scale

        u[i+1] = y[0]

    norm = np.sqrt(np.trapezoid(u**2, r))

    u /= norm

    return r, u

print("\nSearching for s-wave states...")

s_states = search_states(0)

for i,E in enumerate(s_states):
    print(f"  s-state {i+1}: E = {E:.8f} MeV")

print("\nSearching for p-wave states...")

p_states = search_states(1)

for i,E in enumerate(p_states):
    print(f"  p-state {i+2}: E = {E:.8f} MeV")

if len(s_states) > 0:
    E = s_states[0]
    r, u = wavefunction(E, 0)
    plt.figure(figsize=(8,5))
    plt.plot(r, u, label='s-wave')
    plt.xlabel('r (fm)')
    plt.ylabel('u(r)')
    plt.title(f's-wave state, E = {E:.6f} MeV')
    plt.grid()
    plt.legend()
    plt.show()
