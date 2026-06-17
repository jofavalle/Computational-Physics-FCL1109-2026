# =============================================================================
# PARCIAL 3 — REFERENCIA RÁPIDA DE ALGORITMOS
# Mecánica cuántica computacional — sin fcl1109, solo numpy
#
# Usar este archivo como guía de estudio. Memorizar cada bloque antes del
# examen: cubrir el código y reproducirlo desde cero.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# 1. RK4 MANUAL  (lo que sería fc.rk4, pero desde cero)
# ══════════════════════════════════════════════════════════════════════════════
# Sistema de 1er orden: dy/dx = f(x, y),  y ∈ ℝⁿ

def rk4_paso(f, x, y, h):
    k1 = f(x,         y)
    k2 = f(x + h/2,   y + h/2 * k1)
    k3 = f(x + h/2,   y + h/2 * k2)
    k4 = f(x + h,     y + h   * k3)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Uso típico para integrar desde x0 hasta xf:
#   y = y0.copy()
#   for i in range(N - 1):
#       y = rk4_paso(f, x_arr[i], y, h)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BISECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def biseccion(f, a, b, eps=1e-10, Nmax=200):
    """Encuentra raíz de f en [a, b]. Requiere f(a)·f(b) < 0."""
    fa = f(a)
    for _ in range(Nmax):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < eps or 0.5 * abs(b - a) < eps:
            return c
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
    return 0.5 * (a + b)


# ══════════════════════════════════════════════════════════════════════════════
# 3. POZO CUADRADO — ECUACIONES TRASCENDENTES
# ══════════════════════════════════════════════════════════════════════════════
# Unidades: ħ = 1, m = 1/2 → ħ²/2m = 1,  ancho 2a con a = 1
# E_B = −E > 0 es la energía de enlace

def f_par(E_B, V_0):
    """Estados pares: √(V₀−E_B)·tan(√(V₀−E_B)) − √E_B = 0"""
    u = np.sqrt(V_0 - E_B)
    return u * np.tan(u) - np.sqrt(E_B)

def f_impar(E_B, V_0):
    """Estados impares: −√(V₀−E_B)·cot(√(V₀−E_B)) − √E_B = 0"""
    u = np.sqrt(V_0 - E_B)
    return -u / np.tan(u) - np.sqrt(E_B)

# Truco para detectar raíces reales (no singularidades de tan/cot):
# Solo aceptar cambios de signo donde |f(a) − f(b)| < UMBRAL (ej. 50)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHOOTING METHOD — TISE  (clase_04-06-26 → QuantumEigenCall.py)
# ══════════════════════════════════════════════════════════════════════════════
# Ecuación: ψ'' = −(2μ/ħ²)·(E − V(x))·ψ  ≡  −ESCALA·(E−V)·ψ
# Estado:   y = [ψ, ψ'],   y' = [y[1],  −ESCALA·(E−V(x))·y[0]]

# ESCALA = 0.4829 MeV⁻¹·fm⁻² para física nuclear (m_N c² = 940 MeV)

def rhs_schrodinger(x, y, E, V, ESCALA=0.4829):
    return np.array([y[1], -ESCALA * (E - V(x)) * y[0]])

def mismatch(E, h, N_paso, V, ESCALA=0.4829):
    """
    Devuelve (ψ'_L/ψ_L − ψ'_R/ψ_R) / (ψ'_L/ψ_L + ψ'_R/ψ_R)
    en el punto de matching (x = 0, índice i_match = N_paso//3).
    """
    kappa = np.sqrt(-ESCALA * E)   # decaimiento asintótico κ = √(−2μE/ħ²)
    i_match = N_paso // 3

    # ── Onda izquierda: integrar hacia adelante ──
    y = np.array([1.0e-15, kappa * 1.0e-15])
    for ix in range(i_match + 1):
        x = h * (ix - N_paso / 2)
        f_local = lambda x_, y_: rhs_schrodinger(x_, y_, E, V, ESCALA)
        y = rk4_paso(f_local, x, y, h)
    deriv_izq = y[1] / y[0]

    # ── Onda derecha: integrar hacia atrás ──
    y = np.array([1.0e-15, -kappa * 1.0e-15])
    for ix in range(N_paso, i_match, -1):
        x = h * (ix + 1 - N_paso / 2)
        f_local = lambda x_, y_: rhs_schrodinger(x_, y_, E, V, ESCALA)
        y = rk4_paso(f_local, x, y, -h)   # paso negativo
    deriv_der = y[1] / y[0]

    return (deriv_izq - deriv_der) / (deriv_izq + deriv_der)

# Bisección sobre mismatch(E) para encontrar autovalores:
#   Emax = 1.1 * E_prueba;  Emin = E_prueba / 1.1
#   para count in range(Nmax):
#       E_mid = 0.5*(Emax+Emin)
#       if mismatch(Emax)*mismatch(E_mid) > 0: Emax = E_mid
#       else:                                   Emin = E_mid


# ══════════════════════════════════════════════════════════════════════════════
# 5. TDSE — LEAPFROG SPLIT REAL/IMAGINARIO  (clase_09-06-26, Landau 6.10)
# ══════════════════════════════════════════════════════════════════════════════
# ψ = R + iI,  β = Δt/Δx²,   condición de estabilidad: β < 0.5
#
# Esquema (ħ = 1, 2m = 1):
#   R_new[i] = R[i] − β·(I[i+1] + I[i−1] − 2I[i]) + Δt·V[i]·I[i]
#   I_new[i] = I[i] + β·(R_new[i+1] + R_new[i−1] − 2R_new[i]) − Δt·V[i]·R_new[i]
#   CC (pozo infinito): R[0] = R[-1] = I[0] = I[-1] = 0

def tdse_paso(R, I, V, beta, dt):
    """Un paso temporal del leapfrog TDSE."""
    R_new = R.copy()
    I_new = I.copy()
    R_new[1:-1] = (R[1:-1]
                   - beta * (I[2:] + I[:-2] - 2.0 * I[1:-1])
                   + dt * V[1:-1] * I[1:-1])
    R_new[0] = R_new[-1] = 0.0
    I_new[1:-1] = (I[1:-1]
                   + beta * (R_new[2:] + R_new[:-2] - 2.0 * R_new[1:-1])
                   - dt * V[1:-1] * R_new[1:-1])
    I_new[0] = I_new[-1] = 0.0
    return R_new, I_new

# Condición inicial típica (paquete gaussiano con impulso k0):
#   R = exp(−(x−x0)²/2σ²) · cos(k0·x)
#   I = exp(−(x−x0)²/2σ²) · sin(k0·x)
# Normalizar: norma = sqrt(trapz(R² + I², x));  R /= norma; I /= norma


# ══════════════════════════════════════════════════════════════════════════════
# 6. OSCILADOR ARMÓNICO — RK4 + PARIDAD  (Landau Listing 6.1)
# ══════════════════════════════════════════════════════════════════════════════
# TISE: ψ'' = (x² − E)·ψ,   V(x) = x²,   E_n = 2n + 1
# CC asintótica: ψ(x_max) → 0

def rhs_oa(x, y, E):
    """y = [ψ, ψ'],   y' = [ψ', (x² − E)·ψ]"""
    return np.array([y[1], (x**2 - E) * y[0]])

def psi_xmax_oa(E, paridad, x_arr, h):
    """Integra y evalúa ψ(x_max). Paridad: 'par' o 'impar'."""
    y = np.array([1.0, 0.0]) if paridad == 'par' else np.array([0.0, 1.0])
    for x in x_arr[:-1]:
        y = rk4_paso(lambda x_, y_: rhs_oa(x_, y_, E), x, y, h)
    return y[0]

# Energías exactas: E_n = 2n + 1  (n = 0 par, n = 1 impar, n = 2 par, ...)


# ══════════════════════════════════════════════════════════════════════════════
# 7. NORMALIZACIÓN  (siempre al final)
# ══════════════════════════════════════════════════════════════════════════════
# Para una función de onda ψ en una grilla x:
#   norma = np.sqrt(np.trapz(psi**2, x))
#   psi  /= norma


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONDICIONES DE ESTABILIDAD
# ══════════════════════════════════════════════════════════════════════════════
# TDSE leapfrog:  β = Δt/Δx² < 0.5
# Onda (leapfrog clásico): r = c·Δt/Δx ≤ 1
# Calor FTCS: α·Δt/Δx² ≤ 0.5


# ══════════════════════════════════════════════════════════════════════════════
# DEMO RÁPIDO — verificar que los bloques funcionan
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── OAC: primeros 3 estados ──
    x_max = 5.0
    h = 0.01
    x_arr = np.arange(0.0, x_max, h)

    print("Oscilador armónico: autoenergías numéricas vs exactas")
    print(f"{'n':>3}  {'Paridad':>8}  {'E_n (num.)':>12}  {'E_n = 2n+1':>12}")
    E_scan = np.linspace(0.5, 8.0, 3000)
    estados = []
    for paridad in ('par', 'impar'):
        f_vec = [psi_xmax_oa(E, paridad, x_arr, h) for E in E_scan]
        for i in range(len(E_scan) - 1):
            if f_vec[i] * f_vec[i+1] < 0:
                f_local = lambda E, p=paridad: psi_xmax_oa(E, p, x_arr, h)
                En = biseccion(f_local, E_scan[i], E_scan[i+1])
                estados.append((En, paridad))
    estados.sort(key=lambda t: t[0])
    for n, (En, par) in enumerate(estados[:4]):
        print(f"{n:>3}  {par:>8}  {En:>12.6f}  {2*n+1:>12.6f}")

    # ── Bisección simple: √2 ──
    raiz2 = biseccion(lambda x: x**2 - 2, 1.0, 2.0)
    print(f"\nBisección demo: √2 ≈ {raiz2:.10f}  (exacto: {np.sqrt(2):.10f})")
