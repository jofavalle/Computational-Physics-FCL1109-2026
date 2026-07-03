"""
SECCIÓN: LOOPS DE INTEGRACIÓN / ALGORITMOS CENTRALES
======================================================
Los 4 patrones de loop que aparecen en todos los scripts de la unidad 4:

  A. Leapfrog vectorizado  - onda 1D/2D
  B. Loop doble (i,j)      - onda 2D, N-S, Laplace
  C. FTCS vectorizado      - calor/difusión
  D. SOR iterativo         - Laplace, Poisson, N-S

El patrón de "rotar arreglos" es clave: entender y_old → y → y_new.
"""

import numpy as np

# ===========================================================================
# A. LEAPFROG VECTORIZADO - Onda 1D  (lo más rápido de escribir)
# ===========================================================================
# Solo actualiza nodos interiores [1:-1]; los bordes son CC
r2 = (c * dt / dx)**2   # r al cuadrado

for n in range(Nt - 1):
    u_nuevo[1:-1] = (
        2 * u_actual[1:-1]
        - u_anterior[1:-1]
        + r2 * (u_actual[2:] - 2 * u_actual[1:-1] + u_actual[:-2])
    )
    u_nuevo[0] = 0.0     # CC Dirichlet
    u_nuevo[-1] = 0.0

    # rotar: n-1 ← n ← n+1
    u_anterior[:] = u_actual
    u_actual[:]   = u_nuevo


# ===========================================================================
# B1. LEAPFROG CON LOOP for i - Onda 1D  (más legible, más lento)
# ===========================================================================
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        u[n+1, i] = (
            2 * u[n, i]
            - u[n-1, i]
            + r2 * (u[n, i+1] - 2 * u[n, i] + u[n, i-1])
        )
    u[n+1, 0]  = 0.0
    u[n+1, -1] = 0.0
# Nota: guarda TODA la historia en la matriz u[Nt, Nx]


# ===========================================================================
# B2. LOOP DOBLE for i, for j - Membrana 2D
# ===========================================================================
for paso in range(Nt):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u_next[i, j] = (
                2 * u[i, j]
                - u_prev[i, j]
                + r2 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j])
            )
    # CC: bordes fijos
    u_next[0,  :] = 0.0
    u_next[-1, :] = 0.0
    u_next[:,  0] = 0.0
    u_next[:, -1] = 0.0
    # rotar
    u_prev = u.copy()
    u      = u_next.copy()


# ===========================================================================
# B3. LEAPFROG 2D VECTORIZADO (equivalente a B2, más rápido)
# ===========================================================================
for paso in range(Nt):
    u_next[1:-1, 1:-1] = (
        2 * u[1:-1, 1:-1]
        - u_prev[1:-1, 1:-1]
        + r2 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
    )
    u_next[0, :]  = 0.0; u_next[-1, :] = 0.0
    u_next[:, 0]  = 0.0; u_next[:, -1] = 0.0
    u_prev = u.copy()
    u      = u_next.copy()


# ===========================================================================
# C. FTCS VECTORIZADO - Calor/Difusión 1D
# ===========================================================================
r_ftcs = alpha * dt / dx**2    # debe ser ≤ 0.5

for n in range(Nt):
    T_nuevo = T.copy()
    T_nuevo[1:-1] = T[1:-1] + r_ftcs * (T[2:] - 2 * T[1:-1] + T[:-2])
    # CC Dirichlet (los extremos NO se actualizan)
    T_nuevo[0]  = T_izq
    T_nuevo[-1] = T_der
    T = T_nuevo


# ===========================================================================
# D. SOR ITERATIVO - Laplace/Poisson 2D (Gauss-Seidel + relajación)
# ===========================================================================
omega = 1.6          # parámetro de relajación: 1 → Gauss-Seidel, ~1.9 → SOR óptimo
tol   = 1e-4

for it in range(max_iter):
    V_old = V.copy()             # solo para calcular el error (Jacobi)
                                  # en G-S puro no es necesario

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if mask[i, j]:       # saltar placas / obstáculos fijos
                continue
            V_gs = 0.25 * (
                V[i+1, j] + V[i-1, j]
                + V[i, j+1] + V[i, j-1]
                + dx**2 * rho[i, j] / eps0   # = 0 para Laplace puro
            )
            V[i, j] = V[i, j] + omega * (V_gs - V[i, j])

    error = np.max(np.abs(V - V_old))
    if error < tol:
        print(f"Convergió en {it + 1} iteraciones")
        break


# ===========================================================================
# D2. SOR PARA N-S: ψ y ω acopladas
# ===========================================================================
for it in range(max_iter):
    aplicar_frontera()      # CC antes del loop interior

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # --- Poisson para ψ: ∇²ψ = -ω ---
            psi_gs     = 0.25 * (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] + h**2 * w[i,j])
            psi[i, j] += omega * (psi_gs - psi[i, j])

            # --- Transporte de vorticidad estacionario ---
            a1  = w[i+1,j] + w[i-1,j] + w[i,j+1] + w[i,j-1]
            # término convectivo (no lineal):
            conv = (R / 4.0) * (
                (psi[i,j+1] - psi[i,j-1]) * (w[i+1,j] - w[i-1,j])
                - (psi[i+1,j] - psi[i-1,j]) * (w[i,j+1] - w[i,j-1])
            )
            w_gs     = 0.25 * (a1 + conv)
            w[i, j] += omega * (w_gs - w[i, j])

    residual = np.max(np.abs(psi_gs - psi[i, j]))   # simplificado


# ===========================================================================
# LAPLACIANO CON np.roll - Membrana 2D sin loops (clase_28-04-26)
# ===========================================================================
def laplacian(u):
    return (
        np.roll(u,  1, axis=0)    # u[i-1, j]
        + np.roll(u, -1, axis=0)  # u[i+1, j]
        + np.roll(u,  1, axis=1)  # u[i, j-1]
        + np.roll(u, -1, axis=1)  # u[i, j+1]
        - 4 * u
    ) / dx**2
# Uso: u_next = 2*u - u_prev + dt**2 * laplacian(u)
# ⚠ np.roll aplica condiciones PERIÓDICAS. Para Dirichlet, aplicar CC manualmente después.
