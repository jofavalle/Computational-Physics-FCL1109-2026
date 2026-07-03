import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARCIAL 3 - SIMULACRO C
# Tema  : Ecuación de Schrödinger Dependiente del Tiempo (TDSE)
# Método: Leapfrog split real/imaginario (Crank-Nicolson explícito)
# Ref.  : clase_09-06-26, Landau Listings 6.9 / 6.10
# =============================================================================
#
# ENUNCIADO
# ---------
# Simular la evolución temporal de un paquete de onda gaussiano en un pozo
# infinito (caja rígida) de longitud L:
#
#   ψ(x, 0) = N · exp[−(x − x₀)² / (2σ₀²)] · exp(ik₀x)
#
# con condiciones de frontera ψ(0, t) = ψ(L, t) = 0.
#
# La TDSE (con ħ = 1, 2m = 1 → ħ²/2m = 1) es:
#
#   iħ ∂ψ/∂t = −∂²ψ/∂x² + V(x)ψ
#
# Separando ψ = R + iI en parte real e imaginaria:
#
#   ∂R/∂t = −∂²I/∂x² + V·I
#   ∂I/∂t = +∂²R/∂x² − V·R
#
# Esquema leapfrog (Crank-Nicolson explícito, β = Δt/Δx²):
#
#   R_new[i] = R[i] − β·(I[i+1] + I[i−1] − 2I[i]) + Δt·V[i]·I[i]
#   I_new[i] = I[i] + β·(R_new[i+1] + R_new[i−1] − 2R_new[i]) − Δt·V[i]·R_new[i]
#
# CONDICIÓN DE ESTABILIDAD: β = Δt/Δx² < 0.5 (Von Neumann)
#
# ANÁLISIS FÍSICO
# ---------------
# El paquete gaussiano tiene un ancho espectral Δk ~ 1/σ₀. Al propagarse,
# los modos de mayor k avanzan más rápido (relación de dispersión E = k²),
# lo que provoca el ensanchamiento progresivo del paquete (dispersión).
# La probabilidad total ∫|ψ|²dx debe conservarse (sistema conservativo).
# En el pozo infinito, el paquete rebota en las paredes y puede reconstruirse
# parcialmente por interferencia (revivals).
# =============================================================================

# ─── Parámetros de la simulación ─────────────────────────────────────────────
L   = 1.0      # [u.r.] longitud del pozo
Nx  = 500      # número de puntos espaciales
dx  = L / Nx   # [u.r.] paso espacial

dt  = 5.0e-5   # [u.r.] paso temporal
Nt  = 8000     # número de pasos temporales

beta = dt / dx**2    # parámetro de estabilidad
print(f"Parámetro de estabilidad β = Δt/Δx² = {beta:.5f}  (debe ser < 0.5)")
assert beta < 0.5, f"¡Simulación inestable! β = {beta:.4f} ≥ 0.5"

# ─── Grilla espacial ─────────────────────────────────────────────────────────
x = np.linspace(0, L, Nx + 1)   # Nx+1 puntos: x[0] = 0, x[Nx] = L

# ─── Potencial ───────────────────────────────────────────────────────────────
# Pozo infinito: V = 0 adentro (las CC ψ = 0 en los bordes hacen el papel)
V = np.zeros(Nx + 1)

# ─── Condición inicial: paquete gaussiano con impulso k₀ ─────────────────────
x0    = L / 3.0  # [u.r.] posición inicial del paquete
sigma = 0.08     # [u.r.] ancho gaussiano
k0    = 20.0 * np.pi   # [u.r.⁻¹] impulso inicial (vector de onda)

# ψ(x, 0) = A · exp[−(x−x₀)²/2σ²] · exp(ik₀x)
gauss  = np.exp(-0.5 * ((x - x0) / sigma)**2)
R = gauss * np.cos(k0 * x)   # parte real
I = gauss * np.sin(k0 * x)   # parte imaginaria

# Condiciones de frontera
R[0] = R[-1] = 0.0
I[0] = I[-1] = 0.0

# Normalizar: ∫|ψ|²dx = 1
norma = np.sqrt(np.trapz(R**2 + I**2, x))
R /= norma
I /= norma

# ─── Evolución temporal: leapfrog split real/imaginario ──────────────────────
# Guardar snapshots para graficar
instantes_snap  = [0, Nt//4, Nt//2, 3*Nt//4, Nt - 1]
snaps = {}
normas = np.zeros(Nt)

R_new = np.zeros(Nx + 1)
I_new = np.zeros(Nx + 1)

for n in range(Nt):
    # - Paso 1: actualizar R con I del instante anterior -
    R_new[1:-1] = (R[1:-1]
                   - beta * (I[2:] + I[:-2] - 2.0 * I[1:-1])
                   + dt * V[1:-1] * I[1:-1])
    R_new[0] = R_new[-1] = 0.0   # CC

    # - Paso 2: actualizar I con R_new recién calculado -
    I_new[1:-1] = (I[1:-1]
                   + beta * (R_new[2:] + R_new[:-2] - 2.0 * R_new[1:-1])
                   - dt * V[1:-1] * R_new[1:-1])
    I_new[0] = I_new[-1] = 0.0   # CC

    R, I = R_new.copy(), I_new.copy()

    # Conservación de la probabilidad
    normas[n] = np.trapz(R**2 + I**2, x)

    if n in instantes_snap:
        snaps[n] = (R.copy(), I.copy())

# ─── Resultados ──────────────────────────────────────────────────────────────
t_total = Nt * dt
print(f"\nConservación de probabilidad:")
print(f"  t = 0    → ∫|ψ|²dx = {normas[0]:.8f}")
print(f"  t = {t_total/2:.4f} → ∫|ψ|²dx = {normas[Nt//2]:.8f}")
print(f"  t = {t_total:.4f} → ∫|ψ|²dx = {normas[-1]:.8f}")
print(f"  Error máximo: {abs(normas - 1.0).max():.2e}")

# ─── Gráfica ─────────────────────────────────────────────────────────────────
colores = ['navy', 'royalblue', 'steelblue', 'deepskyblue', 'cyan']
t_vals  = np.arange(Nt) * dt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel (0,0): densidad de probabilidad en distintos instantes
for idx, (n_snap, color) in enumerate(zip(instantes_snap[:-1], colores)):
    Rn, In = snaps[n_snap]
    prob = Rn**2 + In**2
    axes[0, 0].plot(x, prob, color=color, lw=2,
                    label=f't = {n_snap * dt:.4f}')
axes[0, 0].set(xlabel='x [u.r.]', ylabel=r'$|\psi(x,t)|^2$',
               title='Densidad de probabilidad - evolución temporal')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel (0,1): parte real en distintos instantes
for idx, (n_snap, color) in enumerate(zip(instantes_snap[:-1], colores)):
    Rn, _ = snaps[n_snap]
    axes[0, 1].plot(x, Rn, color=color, lw=2,
                    label=f't = {n_snap * dt:.4f}')
axes[0, 1].set(xlabel='x [u.r.]', ylabel=r'Re[$\psi(x,t)$]',
               title='Parte real de ψ')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel (1,0): mapa de calor de |ψ(x,t)|²
paso_mapa = max(1, Nt // 300)
t_mapa = np.arange(0, Nt, paso_mapa) * dt
prob_mapa = np.zeros((len(t_mapa), Nx + 1))
# Re-ejecutar para llenar el mapa (usar los datos ya calculados si se guardó todo)
# Aquí usamos una aproximación con los snaps disponibles
R_tmp = snaps[0][0].copy()
I_tmp = snaps[0][1].copy()
R_tmp_new = np.zeros(Nx + 1)
I_tmp_new = np.zeros(Nx + 1)
prob_mapa[0] = R_tmp**2 + I_tmp**2
for j in range(1, len(t_mapa)):
    for _ in range(paso_mapa):
        R_tmp_new[1:-1] = (R_tmp[1:-1]
                           - beta * (I_tmp[2:] + I_tmp[:-2] - 2.0 * I_tmp[1:-1]))
        R_tmp_new[0] = R_tmp_new[-1] = 0.0
        I_tmp_new[1:-1] = (I_tmp[1:-1]
                           + beta * (R_tmp_new[2:] + R_tmp_new[:-2]
                                     - 2.0 * R_tmp_new[1:-1]))
        I_tmp_new[0] = I_tmp_new[-1] = 0.0
        R_tmp, I_tmp = R_tmp_new.copy(), I_tmp_new.copy()
    prob_mapa[j] = R_tmp**2 + I_tmp**2

im = axes[1, 0].imshow(prob_mapa.T, origin='lower', aspect='auto',
                        extent=[0, t_mapa[-1], 0, L],
                        cmap='hot')
axes[1, 0].set(xlabel='t [u.r.]', ylabel='x [u.r.]',
               title=r'Mapa espacio-temporal de $|\psi(x,t)|^2$')
plt.colorbar(im, ax=axes[1, 0])

# Panel (1,1): conservación de probabilidad
axes[1, 1].plot(t_vals, normas, 'k', lw=1.5)
axes[1, 1].axhline(1.0, color='red', linestyle='--', lw=1, label='∫|ψ|²dx = 1')
axes[1, 1].set(xlabel='t [u.r.]', ylabel=r'$\int|\psi|^2\,dx$',
               title='Conservación de la probabilidad',
               ylim=(0.99, 1.01))
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f'TDSE Leapfrog - Paquete gaussiano en pozo infinito'
             f'  ($L={L}$, $x_0={x0}$, $\\sigma={sigma}$, $k_0={k0:.1f}$)',
             fontsize=12)
plt.tight_layout()
plt.show()
