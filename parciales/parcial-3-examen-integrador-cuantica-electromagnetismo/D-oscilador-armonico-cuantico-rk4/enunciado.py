"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES          ║
║                    FÍSICA COMPUTACIONAL  -  FCL1109                          ║
║                    PARCIAL III  ·  SIMULACRO  D                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Eigenstados del Oscilador Armónico Cuántico
  ──────────────────────────────────────────────────────

  El oscilador armónico cuántico (OAC) es uno de los sistemas exactamente
  solubles más importantes de la mecánica cuántica.  En unidades reducidas
  (ħ = m = ω = 1), el Hamiltoniano es:

              H = −d²/dx² + x²

  y la TISE  H ψ = E ψ  se escribe:

              d²ψ/dx²  =  (x² − E) · ψ(x)                       (TISE-OAC)

  Los autovalores exactos son:  E_n = 2n + 1,  con  n = 0, 1, 2, …

  Los autovectores son los polinomios de Hermite × gaussiana:

              ψ_n(x) = C_n · H_n(x) · e^{−x²/2}

  donde H_n es el polinomio de Hermite de grado n.

  ──────────────────────────────────────────────────────────────
  ESTRATEGIA NUMÉRICA
  ──────────────────────────────────────────────────────────────

  Se rescribe la TISE como sistema de primer orden  y = [ψ, ψ']:

              y[0]' = y[1]
              y[1]' = (x² − E) · y[0]                             (*)

  y se integra desde  x = 0  hasta  x = x_max  con RK4.

  Simetría: el potencial V = x² es par, así los autoestados tienen
  paridad definida (par o impar).  Las condiciones iniciales son:

    • Estado par   (n = 0, 2, 4, …): ψ(0) = 1,  ψ'(0) = 0
    • Estado impar (n = 1, 3, 5, …): ψ(0) = 0,  ψ'(0) = 1

  Para un E = E_n correcto, la solución decae como  e^{−x²/2}  para
  x → ∞.  Para E ≠ E_n la solución diverge.  La condición de frontera

              ψ(x_max) = 0      (condición de cuantización)

  se satisface solo para los autovalores. Se aplica bisección sobre la
  función  f(E) = ψ(x_max; E)  para cada paridad por separado.

  Parámetros: x_max = 6,  h = 0.01,  búsqueda en E ∈ [0.5, 12]

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Escriba el sistema de EDOs de primer orden (*) e implemente la función
     rhs(x, y, E)  que lo representa. Implemente un paso de RK4 manual
     rk4_paso(x, y, h, E)  sin utilizar librerías auxiliares del curso.
     ¿Por qué se usa RK4 en lugar de Euler para este problema?

  2. Implemente la función  psi_en_xmax(E, paridad)  que integra desde x = 0
     hasta x = x_max y devuelve ψ(x_max). Utilice las condiciones iniciales
     de paridad descritas arriba.
     Explique físicamente por qué ψ(x_max) debe tender a cero: ¿qué significa
     que una partícula ligada tenga ψ → ∞ para x → ∞?

  3. Escanee E ∈ [0.5, 12] para cada paridad y detecte cambios de signo
     en psi_en_xmax(E).  Para cada cambio de signo encontrado, aplique
     bisección y reporte los primeros 6 autovalores. Compare con la fórmula
     exacta E_n = 2n + 1.

  4. Para cada uno de los primeros 4 estados, re-integre y construya la
     función de onda completa ψ_n(x) en el dominio (−x_max, x_max)
     usando la simetría de paridad:
        - ψ_n(−x) = +ψ_n(x)  para estados pares
        - ψ_n(−x) = −ψ_n(x)  para estados impares
     Normalice cada función de onda.  Grafique ψ_n(x) + E_n (desplazadas
     verticalmente por su energía) junto con el potencial V(x) = x².

  5. Cuente el número de nodos (ceros) de cada ψ_n(x) en el interior.
     ¿Qué relación existe entre el número de nodos y n? Grafique |ψ_n(x)|²
     para n = 0, 1, 2, 3 y discuta:
        - ¿Cuál es la región clásicamente permitida para cada n?
        - ¿Cómo se explica la presencia de ψ_n ≠ 0 fuera de esa región?

  6. Verifique numéricamente la ortogonalidad de los autoestados:

              ∫_{−x_max}^{x_max} ψ_m(x) · ψ_n(x) dx = δ_{mn}

     Calcule la matriz de solapamiento  S[m, n]  para m, n = 0, 1, 2, 3
     (4 × 4) usando  np.trapz.  ¿Qué tan cerca de la identidad es S?
     ¿Por qué la ortogonalidad no es perfecta numéricamente?
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetros ──────────────────────────────────────────────────────────────
x_max      = 6.0    # [u.r.] límite de integración
h          = 0.01   # [u.r.] paso RK4
x_arr      = np.arange(0.0, x_max, h)

E_min_scan = 0.5    # energía mínima de búsqueda
E_max_scan = 12.0   # cubre n = 0..5
N_scan     = 3000
eps        = 1e-10  # tolerancia bisección

# ─── ÍTEM 1: Sistema de EDOs y RK4 ───────────────────────────────────────────

def rhs(x, y, E):
    """y = [ψ, ψ'],  y' = [ψ', (x²−E)·ψ]"""
    # TODO: implementar
    pass

def rk4_paso(x, y, h, E):
    # TODO: implementar k1, k2, k3, k4 y la actualización
    pass

# ─── ÍTEM 2: ψ(x_max) como función de E ─────────────────────────────────────

def psi_en_xmax(E, paridad):
    """
    Integra desde x=0 hasta x_max y devuelve ψ(x_max).
    paridad: 'par' → y = [1, 0];  'impar' → y = [0, 1]
    """
    # TODO: inicializar y según paridad, integrar, devolver y[0]
    pass

# ─── ÍTEM 3: Búsqueda de autovalores ─────────────────────────────────────────
E_scan = np.linspace(E_min_scan, E_max_scan, N_scan)

estados = []   # lista de (E_n, paridad)

for paridad in ('par', 'impar'):
    # TODO: calcular psi_en_xmax para cada E en E_scan,
    #       detectar cambios de signo, refinar con bisección
    pass

estados.sort(key=lambda t: t[0])

print(f"{'n':>3}  {'Paridad':>8}  {'E_n (num.)':>13}  {'E_n = 2n+1':>12}  {'Error':>10}")
for n, (En, par) in enumerate(estados[:6]):
    print(f"{n:>3}  {par:>8}  {En:>13.8f}  {2*n+1:>12.8f}  {abs(En-(2*n+1)):>10.2e}")

# ─── ÍTEM 4: Funciones de onda normalizadas ───────────────────────────────────
# TODO: re-integrar cada estado, reflejar por paridad, normalizar,
#       graficar ψ_n(x) + E_n junto con V(x) = x²

# ─── ÍTEM 5: Densidades y nodos ──────────────────────────────────────────────
# TODO: graficar |ψ_n|², contar nodos, marcar regiones clásicas

# ─── ÍTEM 6: Ortogonalidad ───────────────────────────────────────────────────
# TODO: construir matriz S[m,n] = ∫ψ_m·ψ_n dx para m,n=0..3,
#       imprimir la matriz y discutir su cercanía a la identidad

plt.show()
