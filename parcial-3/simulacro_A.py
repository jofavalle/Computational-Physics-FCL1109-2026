import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARCIAL 3 — SIMULACRO A
# Tema  : Pozo Cuadrado Finito Simétrico en 1D
# Método: Ecuaciones trascendentes + Bisección
# Ref.  : clase_02-06-26, Landau Cap. 6
# =============================================================================
#
# ENUNCIADO
# ---------
# Se tiene un pozo cuadrado finito simétrico de profundidad V₀ y ancho 2a,
# centrado en el origen:
#
#       V(x) = -V₀   para |x| ≤ a
#       V(x) =  0    para |x| > a
#
# En unidades reducidas (ħ = 1, m = 1/2 → ħ²/2m = 1, a = 1):
#
#   — Estados pares:   √(V₀ − Eₗ) · tan(√(V₀ − Eₗ)) = √(Eₗ)
#   — Estados impares: −√(V₀ − Eₗ) · cot(√(V₀ − Eₗ)) = √(Eₗ)
#
# donde Eₗ = −E > 0 es la energía de enlace.
#
# ANÁLISIS FÍSICO
# ---------------
# Las ecuaciones trascendentes emergen de exigir que la función de onda y su
# derivada sean continuas en x = ±a. Esto equivale a igualar las derivadas
# logarítmicas ψ'/ψ en cada lado del borde del pozo.
# Solo ciertos valores de Eₗ satisfacen ambas condiciones simultáneamente →
# espectro discreto. El número de estados ligados aumenta con V₀.
# =============================================================================

# ─── Parámetros del pozo ────────────────────────────────────────────────────
V_0 = 30.0   # [u.r.] profundidad del pozo

# ─── Funciones trascendentes ─────────────────────────────────────────────────

def f_par(E_B):
    """Condición de cuantización para estados pares (simétricos)."""
    u = np.sqrt(V_0 - E_B)
    return u * np.tan(u) - np.sqrt(E_B)

def f_impar(E_B):
    """Condición de cuantización para estados impares (antisimétricos)."""
    u = np.sqrt(V_0 - E_B)
    return -u / np.tan(u) - np.sqrt(E_B)

# ─── Bisección ───────────────────────────────────────────────────────────────

def biseccion(f, a, b, eps=1e-10, Nmax=200):
    """Encuentra una raíz de f en [a, b] por bisección."""
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

# ─── Búsqueda de raíces por cambio de signo ──────────────────────────────────
# Se escanea E_B ∈ (0, V_0). Los saltos grandes en f indican singularidades
# de tan/cot (no raíces reales) y se descartan con el umbral de amplitud.

N_scan = 20000
E_scan = np.linspace(1e-6, V_0 - 1e-6, N_scan)
UMBRAL_SALTO = 50.0   # descartar cambios de signo de singularidades

raices_par   = []
raices_impar = []

for i in range(N_scan - 1):
    a_s, b_s = E_scan[i], E_scan[i+1]

    fp_a, fp_b = f_par(a_s), f_par(b_s)
    if fp_a * fp_b < 0 and abs(fp_a - fp_b) < UMBRAL_SALTO:
        raices_par.append(biseccion(f_par, a_s, b_s))

    fi_a, fi_b = f_impar(a_s), f_impar(b_s)
    if fi_a * fi_b < 0 and abs(fi_a - fi_b) < UMBRAL_SALTO:
        raices_impar.append(biseccion(f_impar, a_s, b_s))

# ─── Resultados ──────────────────────────────────────────────────────────────
# Ordenar de mayor E_B (estado más ligado = estado base n=0) a menor E_B
todos = sorted([(eb, 'par') for eb in raices_par] +
               [(eb, 'impar') for eb in raices_impar],
               key=lambda x: -x[0])   # descendente en E_B = ascendente en E

print(f"Pozo cuadrado finito  V₀ = {V_0}")
print(f"{'n':>3}  {'Paridad':>8}  {'E_B [u.r.]':>14}  {'E = -E_B [u.r.]':>17}")
for n, (eb, paridad) in enumerate(todos):
    print(f"{n:>3}  {paridad:>8}  {eb:>14.8f}  {-eb:>17.8f}")

# ─── Gráfica: funciones trascendentes y sus raíces ───────────────────────────
E_plot = np.linspace(1e-4, V_0 - 1e-4, 8000)
sqrt_E = np.sqrt(E_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel izquierdo — estados pares
fp_plot = np.where(np.abs(f_par(E_plot)) < 25, f_par(E_plot), np.nan)
axes[0].plot(E_plot, fp_plot + sqrt_E, 'b',   lw=2,
             label=r'$\sqrt{V_0-E_B}\,\tan(\sqrt{V_0-E_B})$')
axes[0].plot(E_plot, sqrt_E,            'r--', lw=2,
             label=r'$\sqrt{E_B}$')
for eb in raices_par:
    axes[0].axvline(eb, color='gray', linestyle=':', alpha=0.6)
    axes[0].plot(eb, np.sqrt(eb), 'ro', markersize=9,
                 label=f'$E_B$ = {eb:.3f}')
axes[0].set(xlim=(0, V_0), ylim=(-2, 15),
            xlabel=r'$E_B$ [u.r.]', ylabel='', title='Estados pares (simétricos)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel derecho — estados impares
fi_plot = np.where(np.abs(f_impar(E_plot)) < 25, f_impar(E_plot), np.nan)
axes[1].plot(E_plot, fi_plot + sqrt_E, 'g',   lw=2,
             label=r'$-\sqrt{V_0-E_B}\,\cot(\sqrt{V_0-E_B})$')
axes[1].plot(E_plot, sqrt_E,            'r--', lw=2,
             label=r'$\sqrt{E_B}$')
for eb in raices_impar:
    axes[1].axvline(eb, color='gray', linestyle=':', alpha=0.6)
    axes[1].plot(eb, np.sqrt(eb), 'rs', markersize=9,
                 label=f'$E_B$ = {eb:.3f}')
axes[1].set(xlim=(0, V_0), ylim=(-2, 15),
            xlabel=r'$E_B$ [u.r.]', ylabel='', title='Estados impares (antisimétricos)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle(f'Pozo cuadrado finito — Energías ligadas  ($V_0 = {V_0}$, $a = 1$)',
             fontsize=13)
plt.tight_layout()
plt.show()
