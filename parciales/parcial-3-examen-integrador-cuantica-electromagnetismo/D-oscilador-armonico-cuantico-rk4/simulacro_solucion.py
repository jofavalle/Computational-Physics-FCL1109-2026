import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARCIAL 3 - SIMULACRO D
# Tema  : Oscilador Armónico Cuántico (OAC)
# Método: RK4 manual + Condición de paridad + Bisección en energía
# Ref.  : Landau Listing 6.1 (HOnumeric.py)
# =============================================================================
#
# ENUNCIADO
# ---------
# Resolver la TISE para el oscilador armónico cuántico:
#
#   d²ψ/dx² = (x² − E) · ψ(x)
#
# (Unidades: ħ = m = ω = 1; energía en unidades de ħω; potencial V = x².)
#
# Las energías exactas son: E_n = 2n + 1  (n = 0, 1, 2, ...)
#
# ALGORITMO
# ----------
# 1. Explotar la simetría: estados pares tienen ψ(0) = 1, ψ'(0) = 0;
#    estados impares tienen ψ(0) = 0, ψ'(0) = 1.
# 2. Integrar con RK4 desde x = 0 hasta x = x_max.
# 3. La condición de frontera en el infinito exige ψ(x_max) → 0.
#    La función de mismatch es: f(E) = ψ(x_max) para el estado actual.
# 4. Buscar cambios de signo de f(E) y refinar con bisección.
#
# ANÁLISIS FÍSICO
# ---------------
# Solo para E = E_n los auto-estados decaen exponencialmente en ±∞.
# Para E ≠ E_n, la solución diverge (crece sin límite) porque el término
# (x² − E) > 0 para x grande produce un comportamiento divergente (no físico).
# El número de nodos de ψ(x) coincide exactamente con n (número cuántico).
# El principio de incertidumbre: ⟨x⟩ = 0, Δx·Δp = ħ(n + ½).
# =============================================================================

# ─── Parámetros ──────────────────────────────────────────────────────────────
x_max  = 6.0    # [u.r.] límite de integración (ψ ~ 0 para x > x_max)
h      = 0.01   # [u.r.] paso de integración
x_arr  = np.arange(0.0, x_max, h)   # grilla

E_min_scan = 0.1    # energía mínima de búsqueda
E_max_scan = 12.0   # energía máxima de búsqueda (cubre n = 0..5)
N_scan     = 5000   # puntos del escaneo inicial
eps        = 1e-10  # tolerancia de bisección

# ─── RHS de la TISE: y = [ψ, ψ'] ─────────────────────────────────────────────
def rhs(x, y, E):
    """
    Sistema de 1er orden:  y[0]' = y[1],  y[1]' = (x² − E) · y[0]
    Corresponde a: −ψ'' + x² ψ = E ψ  (en unidades ħ = m = ω = 1)
    """
    return np.array([y[1], (x**2 - E) * y[0]])

# ─── RK4 manual ──────────────────────────────────────────────────────────────
def rk4_paso(x, y, h, E):
    k1 = rhs(x,         y,             E)
    k2 = rhs(x + h/2,   y + h/2 * k1, E)
    k3 = rhs(x + h/2,   y + h/2 * k2, E)
    k4 = rhs(x + h,     y + h   * k3, E)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ─── Integrar y evaluar ψ(x_max) para un E y paridad dados ───────────────────
def psi_en_xmax(E, paridad):
    """
    Integra ψ desde x = 0 hasta x_max.
    paridad = 'par'   → ψ(0) = 1, ψ'(0) = 0
    paridad = 'impar' → ψ(0) = 0, ψ'(0) = 1
    Devuelve ψ(x_max).
    """
    if paridad == 'par':
        y = np.array([1.0, 0.0])
    else:
        y = np.array([0.0, 1.0])
    for x in x_arr[:-1]:
        y = rk4_paso(x, y, h, E)
    return y[0]   # ψ(x_max)

# ─── Bisección para refinar la energía ───────────────────────────────────────
def biseccion(f, a, b, eps=eps, Nmax=200):
    fa = f(a)
    for _ in range(Nmax):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < eps or 0.5 * abs(b - a) < eps:
            return c
        if fa * fc < 0:
            b = c
        else:
            a = c; fa = fc
    return 0.5 * (a + b)

# ─── Búsqueda de autoenergías ─────────────────────────────────────────────────
E_scan = np.linspace(E_min_scan, E_max_scan, N_scan)

autoenergias = []  # lista de (E_n, paridad)

for paridad in ('par', 'impar'):
    f_vec = [psi_en_xmax(E, paridad) for E in E_scan]
    for i in range(len(E_scan) - 1):
        if f_vec[i] * f_vec[i+1] < 0:
            f_local = lambda E, p=paridad: psi_en_xmax(E, p)
            E_n = biseccion(f_local, E_scan[i], E_scan[i+1])
            autoenergias.append((E_n, paridad))

autoenergias.sort(key=lambda x: x[0])

print(f"{'n':>3}  {'Paridad':>8}  {'E_n (num.)':>13}  {'E_n = 2n+1 (exacto)':>21}  {'Error':>10}")
for n, (En, par) in enumerate(autoenergias):
    E_exacto = 2*n + 1
    print(f"{n:>3}  {par:>8}  {En:>13.8f}  {E_exacto:>21.8f}  {abs(En - E_exacto):>10.2e}")

# ─── Graficar los primeros N_plot estados ─────────────────────────────────────
N_plot = min(4, len(autoenergias))   # graficar los primeros 4 estados
colores = ['navy', 'royalblue', 'steelblue', 'deepskyblue']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: funciones de onda ψ_n(x)
for n_estado, (En, par) in enumerate(autoenergias[:N_plot]):
    y = np.array([1.0, 0.0]) if par == 'par' else np.array([0.0, 1.0])
    psi = [y[0]]
    for x in x_arr[:-1]:
        y = rk4_paso(x, y, h, En)
        psi.append(y[0])
    psi = np.array(psi)

    # Reflejar para graficar en (−x_max, x_max)
    x_full = np.concatenate([-x_arr[::-1][:-1], x_arr])
    if par == 'par':
        psi_full = np.concatenate([psi[::-1][:-1], psi])
    else:
        psi_full = np.concatenate([-psi[::-1][:-1], psi])

    # Normalizar
    norma = np.sqrt(np.trapz(psi_full**2, x_full))
    psi_full /= norma

    # Desplazar verticalmente para mejor visualización
    desplaz = En
    axes[0].plot(x_full, psi_full + desplaz, color=colores[n_estado], lw=2,
                 label=f'$\\psi_{n_estado}$  ($E_{n_estado}$ = {En:.3f})')
    axes[0].axhline(desplaz, color=colores[n_estado], linestyle='--', alpha=0.4, lw=1)

# Graficar potencial
x_pot = np.linspace(-x_max + 0.5, x_max - 0.5, 300)
axes[0].plot(x_pot, x_pot**2, 'k', lw=2, alpha=0.3, label='$V(x) = x^2$')
axes[0].set(xlabel='x [u.r.]', ylabel=r'$\psi_n(x) + E_n$',
            title='Funciones de onda del OAC (desplazadas por E_n)',
            xlim=(-x_max + 0.5, x_max - 0.5))
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Panel derecho: densidades de probabilidad |ψ_n|²
for n_estado, (En, par) in enumerate(autoenergias[:N_plot]):
    y = np.array([1.0, 0.0]) if par == 'par' else np.array([0.0, 1.0])
    psi = [y[0]]
    for x in x_arr[:-1]:
        y = rk4_paso(x, y, h, En)
        psi.append(y[0])
    psi = np.array(psi)
    x_full = np.concatenate([-x_arr[::-1][:-1], x_arr])
    if par == 'par':
        psi_full = np.concatenate([psi[::-1][:-1], psi])
    else:
        psi_full = np.concatenate([-psi[::-1][:-1], psi])
    norma = np.sqrt(np.trapz(psi_full**2, x_full))
    psi_full /= norma

    axes[1].plot(x_full, psi_full**2, color=colores[n_estado], lw=2,
                 label=f'$|\\psi_{n_estado}|^2$  (n={n_estado}, {n_estado} nodos)')

axes[1].set(xlabel='x [u.r.]', ylabel=r'$|\psi_n(x)|^2$',
            title='Densidades de probabilidad del OAC',
            xlim=(-x_max + 0.5, x_max - 0.5))
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Oscilador Armónico Cuántico - RK4 + Bisección en energía', fontsize=13)
plt.tight_layout()
plt.show()
