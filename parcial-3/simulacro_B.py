import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARCIAL 3 — SIMULACRO B
# Tema  : Ecuación de Schrödinger 1D — Método de Disparo (Shooting Method)
# Método: RK4 manual + Derivada logarítmica + Bisección
# Ref.  : clase_04-06-26 (QuantumEigenCall.py adaptado), Landau Listing 6.3
# =============================================================================
#
# ENUNCIADO
# ---------
# Encontrar las energías ligadas y funciones de onda del deuterón modelando el
# potencial núcleon-núcleon como un pozo cuadrado:
#
#       V(r) = -V₀  para |r| < R
#       V(r) =  0   para |r| ≥ R
#
# con V₀ = 16 MeV, R = 10 fm (caso simétrico, coordenada r = x).
# Constante de escala: 2μ/ħ² = m_N/(ħc)² ≈ 0.4829 MeV⁻¹·fm⁻²
#
# ALGORITMO: Método de disparo con derivada logarítmica
# ------------------------------------------------------
# 1. Para un E propuesto, integrar ψ_L(x) desde x = -x_max hasta x_match = 0
#    con condición inicial ψ(−x_max) ~ e^{+κ|x|}, κ = √(−2μE/ħ²).
# 2. Integrar ψ_R(x) desde x = +x_max hasta x_match = 0 (retrocediendo).
# 3. Calcular la función de mismatch:
#       f(E) = [ψ'_L / ψ_L − ψ'_R / ψ_R] / [ψ'_L / ψ_L + ψ'_R / ψ_R]
#    Si f(E) = 0 → E es un autovalor (ψ y ψ' son continuas en x_match).
# 4. Aplicar bisección sobre f(E) dentro del intervalo de búsqueda.
#
# ANÁLISIS FÍSICO
# ---------------
# La continuidad de la derivada logarítmica garantiza la conservación de
# la corriente de probabilidad J = −(ħ/m) Im(ψ* ∂ψ/∂x) en el punto de unión.
# Esto es equivalente a exigir que ψ y ψ' sean continuas, condición que
# impone la cuantización del espectro.
# =============================================================================

# ─── Constantes y parámetros ─────────────────────────────────────────────────
ESCALA = 0.4829   # 2m_N/ħ² [MeV⁻¹·fm⁻²]  (m_N c² = 940 MeV, ħc = 197.33 MeV·fm)
V_0    = 16.0     # [MeV]  profundidad del pozo
R_pozo = 10.0     # [fm]   radio del pozo

E_inicial = -17.0            # [MeV] energía de prueba inicial (estado ligado esperado)
E_max     =  1.1 * E_inicial # [MeV] límite superior del intervalo de bisección
E_min     =  E_inicial / 1.1 # [MeV] límite inferior

h      = 0.04    # [fm]   paso de integración
N_paso = 501     # número de pasos totales
eps    = 1e-2    # tolerancia de mismatch

# ─── Potencial ───────────────────────────────────────────────────────────────

def V(x):
    """Pozo cuadrado: −V₀ dentro, 0 fuera."""
    if abs(x) < R_pozo:
        return -V_0
    return 0.0

# ─── Sistema de EDOs: -d²ψ/dx² = ESCALA·(E - V(x))·ψ ───────────────────────
# Estado: y = [ψ, ψ']   →   y' = [ψ', ESCALA·(V(x)−E)·ψ]

def rhs(x, y, E):
    """Lado derecho del sistema de 1er orden para la TISE."""
    dydx = np.zeros(2)
    dydx[0] = y[1]
    dydx[1] = -ESCALA * (E - V(x)) * y[0]
    return dydx

# ─── RK4 manual (sin fcl1109) ────────────────────────────────────────────────

def rk4_paso(x, y, h, E):
    """Un paso de Runge-Kutta de 4to orden."""
    k1 = rhs(x,           y,             E)
    k2 = rhs(x + h/2,     y + h/2 * k1, E)
    k3 = rhs(x + h/2,     y + h/2 * k2, E)
    k4 = rhs(x + h,       y + h   * k3, E)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ─── Función de mismatch ─────────────────────────────────────────────────────

def mismatch(E):
    """
    Integra ψ_L y ψ_R y devuelve la diferencia relativa de derivadas
    logarítmicas en el punto de matching (x = 0).
    """
    i_match = N_paso // 3    # índice del punto de matching

    # Onda izquierda: integrar hacia adelante desde x_min → x_match
    kappa = np.sqrt(-ESCALA * E)     # decaimiento asintótico
    y = np.zeros(2)
    y[0] = 1.0e-15
    y[1] = kappa * y[0]              # ψ_L crece hacia el interior del pozo
    for ix in range(i_match + 1):
        x = h * (ix - N_paso / 2)
        y = rk4_paso(x, y, h, E)
    deriv_izq = y[1] / y[0]         # ψ'_L / ψ_L en x_match

    # Onda derecha: integrar hacia atrás desde x_max → x_match
    y[0] = 1.0e-15
    y[1] = -kappa * y[0]            # ψ_R decae hacia el interior
    for ix in range(N_paso, i_match, -1):
        x = h * (ix + 1 - N_paso / 2)
        y = rk4_paso(x, y, -h, E)  # paso negativo = integrar hacia la izquierda
    deriv_der = y[1] / y[0]         # ψ'_R / ψ_R en x_match

    return (deriv_izq - deriv_der) / (deriv_izq + deriv_der)

# ─── Bisección sobre E ───────────────────────────────────────────────────────

f_max = mismatch(E_max)

print("Búsqueda del autovalor de energía por bisección:")
print(f"{'Iter':>5}  {'E [MeV]':>14}  {'f(E)':>12}")

for iteracion in range(100):
    E_mid  = 0.5 * (E_max + E_min)
    f_mid  = mismatch(E_mid)
    f_max_actual = mismatch(E_max)

    print(f"{iteracion:>5}  {E_mid:>14.7f}  {f_mid:>12.6e}")

    if abs(f_mid) < eps:
        break

    if f_max_actual * f_mid > 0:
        E_max = E_mid
        f_max = f_mid
    else:
        E_min = E_mid

E_final = E_mid
print(f"\nAutovalor convergido: E = {E_final:.6f} MeV")

# ─── Re-integrar para graficar ψ(x) normalizada ──────────────────────────────
N_plot  = 1501
i_match = 500
kappa   = np.sqrt(-ESCALA * E_final)

# Onda izquierda
yL = np.zeros((2, i_match + 1))
y  = np.zeros(2)
y[0] = 1.0e-40
y[1] = kappa * y[0]
for ix in range(i_match + 1):
    yL[0, ix] = y[0]
    yL[1, ix] = y[1]
    x = h * (ix - N_plot / 2)
    y = rk4_paso(x, y, h, E_final)

# Onda derecha
xR_list, psi_R = [], []
y[0] =  1.0e-15
y[1] = -kappa * y[0]
for ix in range(N_plot - 1, i_match + 1, -1):
    x = h * (ix + 1 - N_plot / 2)
    y = rk4_paso(x, y, -h, E_final)
    xR_list.append(x)
    psi_R.append(y[0])

# Normalizar onda izquierda para que coincida con la derecha en x_match
factor_norm = y[0] / yL[0, i_match]
xL_list = [h * (ix - N_plot / 2 + 1) for ix in range(i_match + 1)]
psi_L   = [yL[0, ix] * factor_norm for ix in range(i_match + 1)]

# ─── Gráfica ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: función de onda
axes[0].plot(xL_list, psi_L, 'b', lw=2, label=r'$\psi_L(x)$')
axes[0].plot(xR_list, psi_R, 'r', lw=2, label=r'$\psi_R(x)$')
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5, label='x = 0 (matching)')
axes[0].axvspan(-R_pozo, R_pozo, alpha=0.08, color='blue', label=f'Pozo: |x| < {R_pozo} fm')
axes[0].set(xlabel='x [fm]', ylabel=r'$\psi(x)$',
            title=f'Función de onda  (E = {E_final:.4f} MeV)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Panel 2: densidad de probabilidad
x_total  = xL_list + xR_list[::-1]
psi_total = psi_L  + psi_R[::-1]
prob = np.array(psi_total)**2
norm = np.trapz(prob, x_total)
prob /= norm

axes[1].fill_between(x_total, prob, alpha=0.4, color='steelblue')
axes[1].plot(x_total, prob, 'steelblue', lw=2)
axes[1].axvspan(-R_pozo, R_pozo, alpha=0.08, color='blue')
axes[1].set(xlabel='x [fm]', ylabel=r'$|\psi(x)|^2$',
            title='Densidad de probabilidad normalizada')
axes[1].grid(True, alpha=0.3)

fig.suptitle(f'Pozo nuclear cuadrado  $V_0 = {V_0}$ MeV, $R = {R_pozo}$ fm'
             f' → $E$ = {E_final:.4f} MeV', fontsize=13)
plt.tight_layout()
plt.show()
