"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES          ║
║                    FÍSICA COMPUTACIONAL  -  FCO4101                          ║
║                    PARCIAL III  ·  SIMULACRO  B                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Estados Ligados del Deuterón por el Método de Disparo
  ──────────────────────────────────────────────────────────────────

  Se modela la interacción núcleon-núcleon (potencial del deuterón) como
  un pozo cuadrado unidimensional:

             ⎧ −V₀   si |x| < R
      V(x) = ⎨
             ⎩  0    si |x| ≥ R

  con V₀ = 16 MeV  y  R = 10 fm.  La ecuación de Schrödinger radial es:

               d²ψ/dx² = −(2μ/ħ²)(E − V(x)) ψ(x)

  donde la constante de escala nuclear vale:

               2μ/ħ² ≡ m_N/(ħc)² = 0.4829  MeV⁻¹·fm⁻²

  (con  m_N c² = 940 MeV  y  ħc = 197.33 MeV·fm).

  Se rescribe como sistema de primer orden con  y = [ψ, ψ']:

               y[0]' = y[1]
               y[1]' = −(2μ/ħ²)·(E − V(x))·y[0]              (*)

  ──────────────────────────────────────────────────────────────
  ALGORITMO: Método de Disparo con Derivada Logarítmica
  ──────────────────────────────────────────────────────────────

  Para un valor de energía E propuesto:
    1. Integrar ψ_L desde x_min hasta x_match = 0  (de izquierda a derecha).
    2. Integrar ψ_R desde x_max hasta x_match = 0  (de derecha a izquierda).
    3. Calcular la función de mismatch:

               f(E) = [ψ'_L/ψ_L − ψ'_R/ψ_R] / [ψ'_L/ψ_L + ψ'_R/ψ_R]

       Si f(E) = 0, la derivada logarítmica es continua en x_match,
       lo que garantiza continuidad de ψ y ψ' (autovalor encontrado).
    4. Aplicar bisección sobre f(E) en el intervalo de búsqueda.

  Condiciones de frontera asintóticas:
    - Para x → ±∞ (E < 0):   ψ ~ e^{∓κ|x|}  con  κ = √(−2μE/ħ²)
    - Condición inicial izquierda: ψ_L(x_min) ≃ ε,  ψ'_L(x_min) = κ·ε
    - Condición inicial derecha:   ψ_R(x_max) ≃ ε,  ψ'_R(x_max) = −κ·ε

  Parámetros numéricos:
    h = 0.04 fm  (paso de integración)
    N = 501      (pasos totales)
    Búsqueda:    E ∈ [E/1.1, 1.1·E]  con E_prueba = −17 MeV

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Implemente la función  rhs(x, y, E)  que devuelve el vector
     [y[1], −(2μ/ħ²)·(E − V(x))·y[0]]  correspondiente al sistema (*).
     Implemente también un paso de Runge-Kutta de 4to orden  rk4_paso(x, y, h, E)
     sin utilizar ninguna librería externa del curso. Escriba explícitamente
     las cuatro pendientes k₁, k₂, k₃, k₄.

  2. Implemente la función  mismatch(E)  que:
        a) Integra ψ_L desde x_min hacia x_match con las condiciones iniciales
           de decaimiento exponencial.
        b) Integra ψ_R desde x_max hacia x_match en sentido inverso (h < 0).
        c) Calcula y devuelve f(E) = (ψ'_L/ψ_L − ψ'_R/ψ_R)/(ψ'_L/ψ_L + ψ'_R/ψ_R).
     Explique por qué la continuidad de la derivada logarítmica implica la
     conservación de la corriente de probabilidad J ∝ Im(ψ* ∂ψ/∂x).

  3. Implemente la bisección sobre  mismatch(E)  para refinar el autovalor.
     Utilice el intervalo inicial [E/1.1, 1.1·E]  con E_prueba = −17 MeV.
     Reporte la energía del estado fundamental del deuterón en MeV.

  4. Con el autovalor E encontrado, re-integre ψ_L y ψ_R con mayor resolución
     (N = 1501, x ∈ [−20 fm, 20 fm]). Normalice ψ_L para que coincida con
     ψ_R en x_match y grafique la función de onda completa.
     ¿Es el estado base simétrico o antisimétrico? ¿Cuántos nodos tiene ψ(x)?

  5. Grafique la densidad de probabilidad |ψ(x)|² normalizada tal que
     ∫|ψ|² dx = 1. Calcule y reporte:
        - La probabilidad de encontrar el nucleón dentro del pozo (|x| < R)
        - La probabilidad de encontrar el nucleón fuera del pozo (|x| ≥ R)
     Comente el resultado en términos del alcance de la fuerza nuclear.

  6. Modifique el potencial para V₀ = 10 MeV y repita la búsqueda.
     ¿Sigue existiendo estado ligado? Si la bisección no converge, explique
     por qué físicamente el pozo es demasiado poco profundo. Estime el valor
     mínimo de V₀ para que exista exactamente un estado ligado en este pozo
     con R = 10 fm (condición de umbral: E_B → 0).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Constantes físicas (nuclear) ────────────────────────────────────────────
ESCALA = 0.4829   # 2m_N/ħ²  [MeV⁻¹·fm⁻²]
V_0    = 16.0     # [MeV]  profundidad del pozo
R_pozo = 10.0     # [fm]   semiancho del pozo

# Parámetros numéricos
h      = 0.04     # [fm]   paso de integración RK4
N_paso = 501      # número de pasos totales
eps    = 1e-2     # tolerancia de convergencia del mismatch

E_prueba = -17.0
E_max    =  1.1 * E_prueba
E_min    =  E_prueba / 1.1

# ─── Potencial ───────────────────────────────────────────────────────────────

def V(x):
    if abs(x) < R_pozo:
        return -V_0
    return 0.0

# ─── ÍTEM 1: Sistema de EDOs y RK4 ───────────────────────────────────────────

def rhs(x, y, E):
    # TODO: devolver np.array([y[1], -ESCALA*(E - V(x))*y[0]])
    pass

def rk4_paso(x, y, h, E):
    # TODO: implementar las 4 pendientes y la actualización
    pass

# ─── ÍTEM 2: Función de mismatch ─────────────────────────────────────────────

def mismatch(E):
    """
    Calcula (ψ'_L/ψ_L − ψ'_R/ψ_R) / (ψ'_L/ψ_L + ψ'_R/ψ_R)
    en el punto de matching x = 0.
    """
    kappa   = np.sqrt(-ESCALA * E)   # decaimiento asintótico
    i_match = N_paso // 3

    # Onda izquierda
    # TODO: inicializar y e integrar desde ix=0 hasta ix=i_match+1
    deriv_izq = None   # TODO: y[1] / y[0]

    # Onda derecha
    # TODO: inicializar y e integrar desde ix=N_paso hasta ix=i_match (h<0)
    deriv_der = None   # TODO: y[1] / y[0]

    return (deriv_izq - deriv_der) / (deriv_izq + deriv_der)

# ─── ÍTEM 3: Bisección sobre mismatch ────────────────────────────────────────

# TODO: bucle de bisección, imprimir E en cada iteración, reportar E final

# ─── ÍTEM 4: Re-integrar y graficar ψ(x) ─────────────────────────────────────

# TODO: re-integrar con N=1501, normalizar ψ_L para empatar con ψ_R,
#       graficar la función de onda completa

# ─── ÍTEM 5: Densidad de probabilidad y probabilidades ───────────────────────

# TODO: graficar |ψ|², calcular P_dentro y P_fuera con np.trapz

# ─── ÍTEM 6: Variación de V₀ y condición de umbral ───────────────────────────

# TODO: cambiar V_0 = 10 y repetir la búsqueda; estimar V₀_mínimo

plt.show()
