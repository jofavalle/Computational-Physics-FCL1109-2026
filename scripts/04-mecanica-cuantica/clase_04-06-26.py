# QuantumEigenCall.py: Finds E & psi via rk4 + bisection

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from numpy import *
import numpy as np, matplotlib.pyplot as plt
from fcl1109 import rk4

# m/(hbarc)**2 = 940MeV/(197.33MeV-fm)**2 = 0.4829
eps = 1e-1; Nsteps = 501; h = 0.04; Nmax = 100  # Params
E = -17.; Emax = 1.1*E; Emin = E/1.1

def f(x, y):                                       # RHS for ODE
    global E
    F = zeros((2), float)
    F[0] = y[1]
    F[1] = -(0.4829)*(E-V(x))*y[0]
    return F

def V(x):                                          # Potential
    if(abs(x) < 10.):
        return (-16.0)
    else:
        return(0.)

def diff(h, E_val):                                # Change in log deriv
    global E
    E = E_val                                      # Sync global E for f(x,y)
    y = zeros((2), float)
    i_match = Nsteps//3                            # Matching radius
    nL = i_match + 1
    y[0] = 1.E-15                                  # Initial left wf
    y[1] = y[0]*sqrt(-E*0.4829)
    for ix in range(0, nL+1):
        x = h * (ix - Nsteps/2)
        y = rk4(f, x, y, h)
    left = y[1]/y[0]                               # Log derivative
    y[0] = 1.E-15                                  # Slope for even; reverse if odd
    y[1] = -y[0]*sqrt(-E*0.4829)                   # Initialize R wf
    for ix in range(Nsteps, nL+1, -1):
        x = h*(ix+1-Nsteps/2)
        y = rk4(f, x, y, -h)
    right = y[1]/y[0]                              # Log derivative
    return (left - right)/(left + right)

def plot(h):                                       # Repeat integrations for plot
    global xL, xR, Rwf, Lwf
    Lwf = []; Rwf = []; xR = []; xL = []
    Nsteps_p = 1501                                # Integration steps
    y  = zeros((2), float)
    yL = zeros((2, 505), float)
    i_match = 500                                  # Matching radius
    nL = i_match + 1
    y[0] = 1.E-40                                  # Initial left wf
    y[1] = sqrt(-E*0.4829) * y[0]
    for ix in range(0, nL+1):
        yL[0][ix] = y[0]
        yL[1][ix] = y[1]
        x = h * (ix - Nsteps_p/2)
        y = rk4(f, x, y, h)
    y[0] = -1.E-15                                 # -slope: even; reverse for odd
    y[1] = -sqrt(-E*0.4829)*y[0]
    for ix in range(Nsteps_p-1, nL+2, -1):        # Right WF
        x = h * (ix+1-Nsteps_p/2)                 # Integrate in
        y = rk4(f, x, y, -h)
        xR.append(x)
        Rwf.append(y[0])
    normL = y[0]/yL[0][nL]
    for ix in range(0, nL+1):                      # Normalize L wf & derivative
        x = h * (ix - Nsteps_p/2 + 1)
        y[0] = yL[0][ix]*normL
        y[1] = yL[1][ix]*normL
        xL.append(x)
        Lwf.append(y[0])

# --- Animación ---
plt.ion()
fig, ax = plt.subplots()
ax.grid()
line_l, = ax.plot([], [], label='Left WF')
line_r, = ax.plot([], [], label='Right WF')
ax.set_xlabel('x')
ax.set_ylabel(r'$\psi(x)$', fontsize=18)
ax.legend()

for count in range(0, Nmax):                       # Main program
    E_mid = (Emax + Emin)/2.                       # Bisec E range
    Diff    = diff(h, E_mid)                       # Eval at midpoint
    diffMax = diff(h, Emax)                        # Eval at Emax (bisection ref)
    E = E_mid                                      # Restore global E to midpoint

    if (diffMax*Diff > 0): Emax = E_mid            # Bisection algor
    else:                  Emin = E_mid

    print(f"Iteration {count:3d},  E = {E_mid:.6f} MeV")

    plot(h)
    line_l.set_data(xL, Lwf)
    line_r.set_data(xR, Rwf)
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f'Iter {count}   E = {E_mid:.4f} MeV')
    plt.draw()
    plt.pause(0.3)

    if (abs(Diff) < eps): break

print(f"Final eigenvalue E = {E}")
print(f"Iterations = {count}, max = {Nmax}")

plt.ioff()
ax.set_title(f'R & L Wavefunctions Matched at x = 0   (E = {E:.4f} MeV)')
plt.show()
