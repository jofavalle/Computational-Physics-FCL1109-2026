"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR — FACULTAD DE CIENCIAS NATURALES          ║
║                    FÍSICA COMPUTACIONAL  —  FCO4101                          ║
║                    PARCIAL III  ·  SIMULACRO  A                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Cuantización en un Pozo Cuadrado Finito Simétrico
  ─────────────────────────────────────────────────────────────

  Se considera una partícula cuántica de masa m encerrada en un pozo de
  potencial unidimensional y simétrico de profundidad V₀ y semiancho a:

             ⎧ −V₀   si |x| ≤ a
      V(x) = ⎨
             ⎩  0    si |x| > a

  En unidades reducidas (ħ = 1, m = 1/2,  a = 1), la ecuación de
  Schrödinger independiente del tiempo (TISE) es:

               d²ψ/dx² + (E − V(x)) ψ = 0

  Al exigir continuidad de ψ y de ψ' en x = ±a, la condición de
  cuantización se reduce a dos ecuaciones trascendentes para la
  energía de enlace  E_B = −E > 0  (con  0 < E_B < V₀):

    • Estados pares  (simétricos):
            √(V₀ − E_B) · tan(√(V₀ − E_B)) = √(E_B)          (1)

    • Estados impares  (antisimétricos):
           −√(V₀ − E_B) · cot(√(V₀ − E_B)) = √(E_B)          (2)

  Las raíces de (1) y (2) son las energías ligadas del sistema.

  Condiciones de frontera:
    — ψ(x) crece como e^{+√(E_B) · x} para x → −∞ (región izquierda)
    — ψ(x) decae como e^{−√(E_B) · x} para x → +∞ (región derecha)
    — ψ(x) oscila como cos(k x) / sin(k x) dentro del pozo, k = √(V₀ − E_B)

  Parámetro del pozo: V₀ = 25  [unidades reducidas]

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Implemente las funciones f_par(E_B) y f_impar(E_B) correspondientes
     a las ecuaciones trascendentes (1) y (2). Explique por qué ambas
     funciones deben ser iguales a cero en los autovalores buscados.

  2. Implemente el algoritmo de bisección para encontrar raíces de una
     función arbitraria f(x) en un intervalo [a, b], dado que f(a)·f(b) < 0.
     Tenga en cuenta que tan(u) y cot(u) tienen singularidades: describa
     cómo distinguir numéricamente una singularidad de una raíz real.

  3. Escanee el intervalo (0, V₀) y encuentre TODAS las energías ligadas del
     sistema. Para cada estado reporte:
        — El número cuántico n (comenzando desde n = 0 en el estado base)
        — La paridad (simétrico / antisimétrico)
        — La energía de enlace E_B
        — La energía propia E = −E_B
     Explique por qué el estado base siempre es simétrico para este potencial.

  4. Grafique simultáneamente:
        a) Las funciones del lado izquierdo de (1) y (2)
        b) La función √(E_B)  [lado derecho común]
        c) Marque con un punto los valores E_B hallados
     ¿Qué ocurre con el número de estados ligados al aumentar V₀?
     ¿Cuántos estados existirían si V₀ = 5?

  5. Para el estado base (n = 0) construya la función de onda completa ψ₀(x)
     en el dominio x ∈ [−3a, 3a]:
        — Dentro del pozo:   ψ₀(x) = A cos(k x)
        — Fuera del pozo:    ψ₀(x) = B e^{−κ|x|},  κ = √(E_B₀)
     Determine A y B de la condición de continuidad en x = a y normalice.
     Grafique ψ₀(x) y |ψ₀(x)|².

  6. Interprete físicamente la densidad de probabilidad |ψ₀(x)|²:
        — ¿Dónde es más probable encontrar la partícula?
        — ¿Puede la partícula estar fuera del pozo? Justifique en términos
          de la mecánica clásica vs. la cuántica (efecto túnel).
        — Estime la probabilidad de encontrar la partícula fuera del pozo
          calculando  P_ext = ∫_{|x|>a} |ψ₀|² dx.
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetro del pozo ───────────────────────────────────────────────────────
V_0 = 25.0   # [u.r.] profundidad del pozo

# ─── ÍTEM 1: Ecuaciones trascendentes ────────────────────────────────────────

def f_par(E_B):
    """Estados pares: √(V₀−E_B)·tan(√(V₀−E_B)) − √E_B = 0"""
    # TODO: implementar
    pass

def f_impar(E_B):
    """Estados impares: −√(V₀−E_B)·cot(√(V₀−E_B)) − √E_B = 0"""
    # TODO: implementar
    pass

# ─── ÍTEM 2: Bisección ───────────────────────────────────────────────────────

def biseccion(f, a, b, eps=1e-10, Nmax=200):
    """Encuentra raíz de f en [a, b]. Requiere f(a)·f(b) < 0."""
    # TODO: implementar
    pass

# ─── ÍTEM 3: Búsqueda de todas las raíces ────────────────────────────────────

N_scan = 20000
E_scan = np.linspace(1e-6, V_0 - 1e-6, N_scan)
UMBRAL_SALTO = 50.0   # distinguir singularidades de raíces reales

raices_par   = []
raices_impar = []

# TODO: escanear E_scan, detectar cambios de signo (filtrando singularidades)
# y refinar con biseccion()

# Ordenar por energía (n=0 = estado base = E_B máxima)
todos = sorted(
    [(eb, 'par') for eb in raices_par] + [(eb, 'impar') for eb in raices_impar],
    key=lambda x: -x[0]
)

print(f"Pozo cuadrado finito  V₀ = {V_0}")
print(f"{'n':>3}  {'Paridad':>8}  {'E_B':>12}  {'E = -E_B':>12}")
for n, (eb, paridad) in enumerate(todos):
    print(f"{n:>3}  {paridad:>8}  {eb:>12.6f}  {-eb:>12.6f}")

# ─── ÍTEM 4: Gráfica de ecuaciones trascendentes ─────────────────────────────
# TODO: graficar f_par, f_impar y √E_B en función de E_B, marcar raíces

# ─── ÍTEM 5: Función de onda del estado base ─────────────────────────────────
# TODO: construir ψ₀(x) por tramos, normalizar, graficar ψ₀ y |ψ₀|²

# ─── ÍTEM 6: Probabilidad exterior ───────────────────────────────────────────
# TODO: calcular P_ext = ∫_{|x|>a} |ψ₀|² dx y comentar el resultado

plt.show()
