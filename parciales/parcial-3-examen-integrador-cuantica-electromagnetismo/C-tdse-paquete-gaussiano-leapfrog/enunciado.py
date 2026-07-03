"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES          ║
║                    FÍSICA COMPUTACIONAL  -  FCO4101                          ║
║                    PARCIAL III  ·  SIMULACRO  C                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Evolución Temporal de un Paquete de Onda Gaussiano
  ─────────────────────────────────────────────────────────────

  Se estudia la dinámica de una partícula cuántica libre confinada en un
  pozo de potencial infinito de longitud L (caja rígida), descrita por la
  Ecuación de Schrödinger Dependiente del Tiempo (TDSE):

              iħ ∂ψ/∂t = − (ħ²/2m) ∂²ψ/∂x² + V(x)ψ

  En unidades reducidas  (ħ = 1, 2m = 1 → ħ²/2m = 1, L = 1, V = 0 adentro):

              i ∂ψ/∂t = −∂²ψ/∂x²                                 (TDSE)

  ──────────────────────────────────────────────────────────────
  ESQUEMA NUMÉRICO: Leapfrog Split Real/Imaginario
  ──────────────────────────────────────────────────────────────

  Se separa  ψ(x, t) = R(x, t) + i·I(x, t)  y se obtiene el sistema:

              ∂R/∂t = −∂²I/∂x² + V·I
              ∂I/∂t = +∂²R/∂x² − V·R

  La discretización leapfrog en dos semi-pasos, con  β = Δt/Δx², da:

    [Paso 1]   R_new[i] = R[i] − β(I[i+1]+I[i−1]−2I[i]) + Δt·V[i]·I[i]
    [Paso 2]   I_new[i] = I[i] + β(R_new[i+1]+R_new[i−1]−2R_new[i]) − Δt·V[i]·R_new[i]

  Nótese que el Paso 2 usa los valores R_new recién calculados en el Paso 1.

  Condición de estabilidad de Von Neumann:  β = Δt/Δx² < 0.5

  Condiciones de frontera (pozo infinito):
              R(0, t) = R(L, t) = 0
              I(0, t) = I(L, t) = 0

  Condición inicial - paquete de onda gaussiano con impulso k₀:

              ψ(x, 0) = N·exp[−(x−x₀)²/(2σ₀²)] · e^{ik₀x}

  descompuesta en:
              R(x, 0) = N·exp[−(x−x₀)²/(2σ₀²)] · cos(k₀x)
              I(x, 0) = N·exp[−(x−x₀)²/(2σ₀²)] · sin(k₀x)

  donde N se determina de  ∫₀ᴸ (R² + I²) dx = 1.

  Parámetros:  L = 1,  Nx = 500,  Δt = 5×10⁻⁵,  Nt = 8000,
               x₀ = L/3,  σ₀ = 0.08,  k₀ = 20π

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Calcule el parámetro de estabilidad  β = Δt/Δx²  con los valores dados
     y verifique que se cumple la condición β < 0.5. Explique qué ocurriría
     físicamente con la simulación si se viola esta condición (β ≥ 0.5):
     ¿qué magnitud diverge primero?

  2. Construya la condición inicial ψ(x, 0) = R(x, 0) + i·I(x, 0) y
     normalícela de modo que  ∫₀ᴸ |ψ(x, 0)|² dx = 1.
     Aplique las condiciones de frontera (CC).
     ¿Qué representa físicamente el factor e^{ik₀x}? ¿En qué dirección
     se propaga el paquete? Justifique en términos del operador de momento.

  3. Implemente la función  tdse_paso(R, I, V, beta, dt)  que realiza un
     paso temporal completo de acuerdo al esquema de dos semi-pasos descrito
     arriba. Preste atención al orden: el Paso 2 debe usar R_new, no R.

  4. Ejecute la simulación durante Nt pasos y guarde instantáneas de |ψ(x,t)|²
     en al menos 4 instantes distintos. Grafique las instantáneas superpuestas
     en un mismo panel. Describa cualitativamente:
        - ¿El paquete se desplaza? ¿En qué dirección y con qué rapidez?
        - ¿El ancho del paquete cambia con el tiempo? Explique este efecto
          usando la relación de dispersión E = k² (unidades reducidas).

  5. En cada paso de tiempo calcule la norma  N(t) = ∫₀ᴸ |ψ(x,t)|² dx.
     Grafique N(t) versus t. ¿Se conserva la probabilidad? ¿Cuál es el
     error absoluto máximo |N(t) − 1|? Relacione este error con el orden
     de precisión del esquema numérico (O(Δt², Δx²)).

  6. Repita la simulación con  β = 0.6  (violando la estabilidad).
     Grafique |ψ(x,t)|² al cabo de pocos pasos. Describa qué observa y
     explique por qué el esquema diverge cuando β > 0.5 (analice el factor
     de amplificación G en el análisis de Von Neumann: |G|² = 1 − 4β(1−β)sin²θ).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetros de la simulación ─────────────────────────────────────────────
L   = 1.0        # [u.r.] longitud del pozo infinito
Nx  = 500        # puntos espaciales
dx  = L / Nx     # paso espacial [u.r.]

dt  = 5.0e-5     # paso temporal [u.r.]
Nt  = 8000       # pasos temporales

x0    = L / 3.0  # posición inicial del paquete
sigma = 0.08     # ancho gaussiano
k0    = 20.0 * np.pi   # impulso inicial [u.r.⁻¹]

# ─── ÍTEM 1: Verificar estabilidad ───────────────────────────────────────────
beta = dt / dx**2
print(f"β = Δt/Δx² = {beta:.5f}  (condición: β < 0.5)")
# TODO: agregar assert y comentar qué pasaría si β ≥ 0.5

# ─── Grilla espacial y potencial ─────────────────────────────────────────────
x = np.linspace(0, L, Nx + 1)   # Nx+1 puntos
V = np.zeros(Nx + 1)             # V = 0 (pozo infinito: CC en bordes)

# ─── ÍTEM 2: Condición inicial ───────────────────────────────────────────────
# TODO: construir R e I a partir del paquete gaussiano
# TODO: aplicar CC: R[0] = R[-1] = I[0] = I[-1] = 0
# TODO: normalizar para que ∫(R²+I²)dx = 1

R = np.zeros(Nx + 1)   # reemplazar con la CI correcta
I = np.zeros(Nx + 1)   # reemplazar con la CI correcta

# ─── ÍTEM 3: Un paso temporal del leapfrog ───────────────────────────────────

def tdse_paso(R, I, V, beta, dt):
    """
    Realiza un paso temporal leapfrog split R/I.
    Paso 1: actualizar R usando I del instante anterior.
    Paso 2: actualizar I usando R_new recién calculado.
    """
    R_new = R.copy()
    I_new = I.copy()

    # Paso 1 - TODO: actualizar R_new[1:-1] y aplicar CC
    # R_new[1:-1] = R[1:-1] − β·(I[i+1]+I[i−1]−2I[i]) + Δt·V·I

    # Paso 2 - TODO: actualizar I_new[1:-1] usando R_new y aplicar CC
    # I_new[1:-1] = I[1:-1] + β·(R_new[i+1]+R_new[i−1]−2R_new[i]) − Δt·V·R_new

    return R_new, I_new

# ─── ÍTEM 4 y 5: Evolución temporal ─────────────────────────────────────────
instantes = [0, Nt//4, Nt//2, 3*Nt//4, Nt - 1]
snaps  = {}
normas = np.zeros(Nt)

for n in range(Nt):
    # TODO: llamar a tdse_paso y actualizar R, I
    # TODO: calcular normas[n] = np.trapz(R**2 + I**2, x)
    if n in instantes:
        snaps[n] = (R.copy(), I.copy())

# ─── ÍTEM 4: Gráfica de instantáneas ─────────────────────────────────────────
# TODO: graficar |ψ(x,t)|² = R²+I² para cada instante guardado

# ─── ÍTEM 5: Conservación de probabilidad ────────────────────────────────────
# TODO: graficar normas vs tiempo, calcular error máximo

# ─── ÍTEM 6: Caso inestable β > 0.5 ─────────────────────────────────────────
# TODO: repetir con dt tal que β = 0.6 y mostrar la divergencia

plt.show()
