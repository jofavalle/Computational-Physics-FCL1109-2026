"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  -  FCL1109                            ║
║                    PARCIAL III  ·  SIMULACRO  A  -  SOLUCIÓN                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Cuantización en un Pozo Cuadrado Finito Simétrico (1D)
  ───────────────────────────────────────────────────────────────

      V(x) = -V₀   si |x| ≤ a          V(x) = 0   si |x| > a

  Unidades reducidas:  ħ = 1,  m = 1/2  (⇒ ħ²/2m = 1),  a = 1.
  (Trabajar en "unidades reducidas" significa elegir las constantes físicas
   iguales a 1 para que no aparezcan en las fórmulas y los números queden limpios.)

  TISE (ecuación de Schrödinger independiente del tiempo):   ψ'' + (E - V(x)) ψ = 0.

  Para un estado LIGADO (E < 0) definimos la energía de enlace
  E_B = -E > 0, con 0 < E_B < V₀.  Las soluciones por regiones son:

      • Dentro  (|x| ≤ a):  ψ'' + (V₀ - E_B) ψ = 0  → oscila, k = √(V₀ - E_B)
      • Fuera   (|x| > a):  ψ'' - E_B ψ = 0          → decae, κ = √(E_B)

  Imponiendo continuidad de ψ y ψ' en x = ±a (igualar la derivada
  logarítmica ψ'/ψ a cada lado) se obtienen las dos ecuaciones de
  cuantización (con a = 1):

      • Pares  (simétricos, ψ ∝ cos kx):   √(V₀-E_B)·tan√(V₀-E_B) = √E_B   (1)
      • Impares(antisim.,   ψ ∝ sin kx):  -√(V₀-E_B)·cot√(V₀-E_B) = √E_B   (2)

  Las raíces de (1) y (2) son las energías ligadas del sistema.

  ESTRATEGIA DEL CÓDIGO: en vez de resolver la ecuación diferencial, usamos las
  ecuaciones (1) y (2) ya deducidas a mano. El problema cuántico se reduce a un
  problema puramente numérico: "encontrar las raíces de una función f(E_B)=0".
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Importar librerías ────────────────────────────────────────────────────────
# 'import X as Y' carga la librería X y le da el apodo corto Y para escribir menos.
import numpy as np               # NumPy: arreglos numéricos y funciones matemáticas
                                 # (np.sqrt, np.sin, np.linspace, ...). Operan tanto
                                 # sobre un número suelto como sobre un arreglo entero.
import matplotlib.pyplot as plt  # Matplotlib: librería para dibujar gráficas.

# ─── Parámetros del pozo ──────────────────────────────────────────────────────
V_0 = 25.0   # [u.r.] profundidad del pozo. El '.0' lo hace flotante (decimal).
a   = 1.0    # [u.r.] semiancho (fijado por las unidades reducidas)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 1 - Ecuaciones trascendentes
# ══════════════════════════════════════════════════════════════════════════════
# ¿Por qué deben anularse en los autovalores?
# -------------------------------------------
# La continuidad de ψ y ψ' en el borde x = a exige que la derivada logarítmica
# ψ'/ψ calculada con la solución interior (oscilante) coincida con la calculada
# con la exterior (decreciente).  Esa igualdad es exactamente la ecuación (1)
# [pares] o (2) [impares].  Si escribimos cada ecuación como  LHS - RHS = 0,
# entonces f(E_B) = 0  ⇔  el empalme es posible  ⇔  E_B es un autovalor físico.
# Para E_B que NO satisface f = 0 la función de onda sería discontinua (o su
# derivada lo sería) y no representa un estado estacionario admisible.

# 'def nombre(argumentos):' DEFINE una función. El texto entre triples comillas
# justo debajo es el "docstring": una descripción que queda asociada a la función.
def f_par(E_B):
    """Estados pares:  √(V₀−E_B)·tan(√(V₀−E_B)) − √E_B = 0."""
    u = np.sqrt(V_0 - E_B)            # u = k = √(V₀ − E_B). np.sqrt = raíz cuadrada.
    return u * np.tan(u) - np.sqrt(E_B)   # 'return' = el valor que devuelve la función.
                                          # np.tan = tangente. Esto es el lado izq. menos
                                          # el derecho de la ec. (1): vale 0 en una raíz.

def f_impar(E_B):
    """Estados impares:  −√(V₀−E_B)·cot(√(V₀−E_B)) − √E_B = 0."""
    u = np.sqrt(V_0 - E_B)            # k = √(V₀ − E_B)
    # cot(u) = 1/tan(u) = cos/sin. NumPy no tiene 'cotangente', así que usamos -u/tan(u).
    return -u / np.tan(u) - np.sqrt(E_B)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 2 - Bisección
# ══════════════════════════════════════════════════════════════════════════════
# La BISECCIÓN es el método más simple para hallar una raíz (un x donde f(x)=0):
# si f cambia de signo entre a y b, hay una raíz en medio. Partimos el intervalo
# por la mitad una y otra vez, quedándonos siempre con la mitad donde sigue
# habiendo cambio de signo. El intervalo se reduce a la mitad en cada paso.
#
# Singularidad vs. raíz real:
# ---------------------------
# tan(u) y cot(u) divergen a ±∞ en sus polos.  Cerca de un polo, f cambia de
# signo saltando de +∞ a −∞ SIN pasar por cero de forma continua: es una
# "raíz falsa".  Numéricamente se distingue porque el salto |f(b) − f(a)| entre
# dos nodos contiguos de la malla es ENORME (la función es continua y suave
# alrededor de una raíz real, pero discontinua alrededor de un polo).  Por eso
# se exige, además del cambio de signo, que |f(b) − f(a)| < UMBRAL_SALTO.

# Esta función recibe OTRA función 'f' como argumento (en Python las funciones se
# pueden pasar como un dato más). 'eps' y 'Nmax' tienen valores por defecto: si no
# se indican al llamar, se usan 1e-12 y 200 (1e-12 = 1×10⁻¹²).
def biseccion(f, a, b, eps=1e-12, Nmax=200):
    """Raíz de f en [a, b] por bisección.  Requiere f(a)·f(b) < 0."""
    fa = f(a)                          # valor de f en el extremo izquierdo
    for _ in range(Nmax):              # repetir hasta Nmax veces. '_' = no usamos el contador.
        c  = 0.5 * (a + b)             # c = punto medio del intervalo
        fc = f(c)                      # valor de f en el punto medio
        # Criterio de parada: o f(c) es casi 0, o el intervalo ya es minúsculo.
        if abs(fc) < eps or 0.5 * (b - a) < eps:
            return c                   # devolvemos la raíz y SALIMOS de la función
        # ¿En qué mitad está la raíz? Donde el producto de signos sea negativo.
        if fa * fc < 0.0:      # f(a) y f(c) tienen signos opuestos → raíz en [a, c]
            b = c                      # achicamos por la derecha
        else:                  # si no, la raíz está en [c, b]
            a, fa = c, fc              # achicamos por la izquierda (y guardamos f(c))
    return 0.5 * (a + b)               # si se agotan las iteraciones, mejor estimación


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 3 - Búsqueda de TODAS las energías ligadas
# ══════════════════════════════════════════════════════════════════════════════
# Idea: la bisección necesita un intervalo donde f cambie de signo. Como no sabemos
# de antemano dónde están las raíces, "barremos" todo el rango (0, V₀) en muchos
# pasos pequeños y, cada vez que f cambia de signo entre dos pasos consecutivos,
# lanzamos la bisección en ese pequeño intervalo.

N_scan       = 40000     # nº de puntos del barrido (más puntos = no perder raíces juntas)
# np.linspace(inicio, fin, N) crea un arreglo de N números igualmente espaciados
# entre 'inicio' y 'fin'. Empezamos en 1e-7 (casi 0) y terminamos casi en V₀ para
# evitar las divisiones por cero / raíces de números negativos en los extremos.
E_scan       = np.linspace(1e-7, V_0 - 1e-7, N_scan)
UMBRAL_SALTO = 50.0      # descarta los cambios de signo por polos de tan/cot (ver ítem 2)

raices_par, raices_impar = [], []   # dos listas vacías donde guardaremos las raíces

# Recorremos el barrido de a pares de puntos consecutivos: (E_scan[i], E_scan[i+1]).
for i in range(N_scan - 1):
    a_s, b_s = E_scan[i], E_scan[i + 1]    # extremos del subintervalo i-ésimo

    fp_a, fp_b = f_par(a_s), f_par(b_s)    # f_par en ambos extremos
    # Hubo cambio de signo (producto < 0) Y no es un polo (salto moderado) → raíz real.
    if fp_a * fp_b < 0.0 and abs(fp_b - fp_a) < UMBRAL_SALTO:
        raices_par.append(biseccion(f_par, a_s, b_s))   # .append añade a la lista

    fi_a, fi_b = f_impar(a_s), f_impar(b_s)
    if fi_a * fi_b < 0.0 and abs(fi_b - fi_a) < UMBRAL_SALTO:
        raices_impar.append(biseccion(f_impar, a_s, b_s))

# Queremos listar los estados ordenados del más ligado (E_B grande) al menos ligado.
# Construimos una sola lista de pares (energía, etiqueta) usando "comprensiones de
# lista": [expresión for elemento in lista] genera un nuevo elemento por cada raíz.
# 'sorted(..., key=...)' ordena; 'key=lambda x: -x[0]' ordena por el 1er elemento
# (la energía) con signo cambiado → de mayor a menor. ('lambda' = función anónima.)
todos = sorted([(eb, 'par')   for eb in raices_par] +
               [(eb, 'impar') for eb in raices_impar],
               key=lambda x: -x[0])

# print(...) imprime en pantalla. Las f-strings (f"...") permiten incrustar valores
# entre llaves {}. ':>3' alinea a la derecha en 3 espacios; '.8f' = 8 decimales.
print("═" * 60)                          # "═"*60 repite el carácter 60 veces
print(f"ÍTEM 3 - Espectro de estados ligados   (V₀ = {V_0},  a = {a})")
print("═" * 60)
print(f"{'n':>3}  {'Paridad':>9}  {'E_B [u.r.]':>14}  {'E = -E_B [u.r.]':>17}")
# enumerate(lista) entrega de a pares (índice, elemento): aquí n=0,1,2,... y el par (eb, paridad).
for n, (eb, paridad) in enumerate(todos):
    print(f"{n:>3}  {paridad:>9}  {eb:>14.8f}  {-eb:>17.8f}")
print(f"\nNúmero total de estados ligados: {len(todos)}")   # len = longitud de la lista
print("(esperado: ⌈2√V₀/π⌉ = ⌈2·5/π⌉ = 4)")

# ── ¿Por qué el estado base es SIEMPRE simétrico (par)? ───────────────────────
# 1) Argumento de nodos: la energía crece con el número de nodos de ψ.  La
#    solución par (∝ cos kx) no tiene nodo en x = 0; la impar (∝ sin kx) sí.
#    El estado de mínima energía es el de MENOS nodos → necesariamente par.
# 2) Argumento de existencia: la ecuación (1) SIEMPRE tiene al menos una raíz
#    (tan parte de 0 y crece), por pequeño que sea V₀; en cambio la ecuación
#    (2) sólo tiene solución si √V₀ > π/2.  Luego siempre existe un estado par,
#    y es el más profundo.
print("\nEl estado base (n=0) es PAR: es la solución sin nodos (mínima energía)\n"
      "y la ec. par siempre admite raíz aunque V₀ sea pequeño.")


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 4 - Gráfica de las ecuaciones trascendentes
# ══════════════════════════════════════════════════════════════════════════════
# Idea visual: dibujar por separado el lado izquierdo (tan/cot) y el lado derecho
# (√E_B) de las ecuaciones. Donde las dos curvas se CRUZAN está una raíz/autovalor.
# Reescribimos (1) y (2) como  g_par(E_B) = √(V₀-E_B)·tan(...)  y
# g_impar(E_B) = -√(V₀-E_B)·cot(...), y los cortes con √E_B son las raíces.
E_plot = np.linspace(1e-4, V_0 - 1e-4, 8000)   # malla fina sólo para dibujar suave
sqrt_E = np.sqrt(E_plot)                        # lado derecho √E_B (NumPy opera sobre TODO el arreglo)
u_plot = np.sqrt(V_0 - E_plot)                  # k(E_B) en cada punto del arreglo

g_par   = u_plot * np.tan(u_plot)               # lado izquierdo de la ec. par
g_impar = -u_plot / np.tan(u_plot)              # lado izquierdo de la ec. impar
# Cerca de los polos tan/cot se disparan a ±∞ y ensucian el dibujo. np.where(cond, A, B)
# elige, elemento a elemento, A donde 'cond' es verdadero y B donde es falso. Aquí:
# si |g| < 25 dejamos g; si no, ponemos np.nan ("no es un número") que Matplotlib
# simplemente NO dibuja → así "cortamos" las ramas verticales de los polos.
g_par   = np.where(np.abs(g_par)   < 25, g_par,   np.nan)
g_impar = np.where(np.abs(g_impar) < 25, g_impar, np.nan)

# plt.subplots(1, 2) crea una figura con 1 fila y 2 columnas de gráficas. Devuelve
# la figura (fig) y un arreglo de ejes (axes): axes[0] es el panel izq., axes[1] el der.
# figsize=(ancho, alto) en pulgadas.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel izquierdo - estados pares
# .plot(x, y, 'b', lw=2, label=...) dibuja la curva y(x). 'b'=azul, lw=grosor de línea.
# label es el texto para la leyenda. La 'r' antes de '...' es una "raw string" que
# evita que Python interprete '\'; sirve para escribir fórmulas LaTeX ($...$).
axes[0].plot(E_plot, g_par, 'b', lw=2, label=r'$\sqrt{V_0-E_B}\,\tan\sqrt{V_0-E_B}$')
axes[0].plot(E_plot, sqrt_E, 'r--', lw=2, label=r'$\sqrt{E_B}$')   # 'r--' = roja a trazos
for eb in raices_par:                                    # marcar cada raíz hallada
    axes[0].axvline(eb, color='gray', ls=':', alpha=0.6) # línea vertical punteada en la raíz
    axes[0].plot(eb, np.sqrt(eb), 'ko', ms=9, zorder=5,  # 'ko' = punto negro; ms=tamaño
                 label=f'$E_B = {eb:.3f}$')
# .set(...) fija varias propiedades del eje de golpe: límites, etiquetas, título.
axes[0].set(xlim=(0, V_0), ylim=(-2, 15), xlabel=r'$E_B$ [u.r.]',
            title='Estados pares (simétricos)')
axes[0].legend(fontsize=8, loc='upper right')   # muestra la leyenda (los 'label')
axes[0].grid(alpha=0.3)                          # cuadrícula tenue (alpha=transparencia)

# Panel derecho - estados impares (misma lógica)
axes[1].plot(E_plot, g_impar, 'g', lw=2, label=r'$-\sqrt{V_0-E_B}\,\cot\sqrt{V_0-E_B}$')
axes[1].plot(E_plot, sqrt_E, 'r--', lw=2, label=r'$\sqrt{E_B}$')
for eb in raices_impar:
    axes[1].axvline(eb, color='gray', ls=':', alpha=0.6)
    axes[1].plot(eb, np.sqrt(eb), 'ks', ms=9, zorder=5,   # 'ks' = cuadrado negro
                 label=f'$E_B = {eb:.3f}$')
axes[1].set(xlim=(0, V_0), ylim=(-2, 15), xlabel=r'$E_B$ [u.r.]',
            title='Estados impares (antisimétricos)')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(alpha=0.3)

fig.suptitle(f'Pozo cuadrado finito - ecuaciones de cuantización '   # título general
             f'($V_0={V_0}$, $a={a}$)', fontsize=13)
plt.tight_layout()    # ajusta los márgenes para que nada se solape

# ── ¿Qué pasa con el nº de estados al aumentar V₀?  ¿Y si V₀ = 5? ─────────────
# El nº de estados ligados ≈ ⌈2√V₀ / π⌉ : crece como √V₀ (cada vez que √V₀
# supera un múltiplo de π/2 aparece un nuevo estado, alternando par/impar).
# Para V₀ = 5:  √5 ≈ 2.236  →  ⌈2·2.236/π⌉ = ⌈1.42⌉ = 2 estados (1 par + 1 impar).
print("\n" + "═" * 60)
print("ÍTEM 4 - Nº de estados ligados  ≈ ⌈2√V₀/π⌉")
print("═" * 60)
for Vtest in (5.0, 25.0, 100.0):                  # probamos tres profundidades
    # np.ceil = redondeo hacia arriba (techo ⌈⌉); int() lo convierte a entero.
    print(f"  V₀ = {Vtest:6.1f}  →  {int(np.ceil(2*np.sqrt(Vtest)/np.pi))} estados")
print("  Crece monótonamente con V₀ (∝ √V₀).  Para V₀ = 5 hay 2 estados.")


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 5 - Función de onda del estado base ψ₀(x)
# ══════════════════════════════════════════════════════════════════════════════
# Ya conocemos las energías; ahora construimos la FORMA de la función de onda del
# estado fundamental (el más ligado) y la normalizamos (área bajo |ψ|² igual a 1).
E_B0 = todos[0][0]                       # todos[0] es el 1er estado (más ligado); [0] toma su energía
k    = np.sqrt(V_0 - E_B0)               # nº de onda DENTRO del pozo (oscilación)
kap  = np.sqrt(E_B0)                     # κ = constante de decaimiento FUERA del pozo

# Forma:  ψ₀(x) = A cos(kx)        |x| ≤ a   (oscila, simétrica)
#         ψ₀(x) = B e^{-κ|x|}      |x| > a   (cola que decae)
# Continuidad en x = a:  A cos(ka) = B e^{-κa}  →  B = A cos(ka) e^{+κa}.
# (La continuidad de ψ' ya está garantizada porque E_B0 resuelve la ec. par.)
#
# Normalización analítica  ∫|ψ₀|² dx = 1:
#   Interior: A² ∫_{-a}^{a} cos²(kx) dx = A² [a + sin(2ka)/(2k)]
#   Exterior: 2 B² ∫_{a}^{∞} e^{-2κx} dx = B² e^{-2κa}/κ = A² cos²(ka)/κ
#   ⇒ A = [ a + sin(2ka)/(2k) + cos²(ka)/κ ]^{-1/2}
A = 1.0 / np.sqrt(a + np.sin(2*k*a)/(2*k) + np.cos(k*a)**2 / kap)   # constante interior
B = A * np.cos(k*a) * np.exp(kap*a)                                 # constante exterior; np.exp = e^()

def psi0(x):
    """Función de onda normalizada del estado base, definida POR TRAMOS."""
    x = np.asarray(x, dtype=float)       # asegura que x sea un arreglo NumPy de flotantes
    dentro = np.abs(x) <= a              # arreglo de True/False: ¿está dentro del pozo?
    # np.where(condición, valor_si_True, valor_si_False), elemento a elemento:
    # dentro usamos A·cos(kx); fuera, B·e^{-κ|x|}. np.abs = valor absoluto.
    return np.where(dentro, A*np.cos(k*x), B*np.exp(-kap*np.abs(x)))

# Verificación numérica de la normalización: integramos |ψ|² y debe dar ≈ 1.
# np.linspace(-3a, 3a, 200001) = malla muy fina de x para integrar con precisión.
x_norm = np.linspace(-3*a, 3*a, 200001)
# np.trapezoid(y, x) aproxima la integral ∫ y dx por la "regla del trapecio"
# (suma de áreas de trapecios bajo la curva). Es integración numérica.
norma  = np.trapezoid(psi0(x_norm)**2, x_norm)
print("\n" + "═" * 60)
print("ÍTEM 5 - Estado base")
print("═" * 60)
print(f"  E_B0 = {E_B0:.8f},  k = {k:.6f},  κ = {kap:.6f}")
print(f"  A = {A:.6f},  B = {B:.6f}")
print(f"  ∫|ψ₀|² dx (numérico) = {norma:.8f}   (debe ≈ 1)")

# Gráfica de ψ₀(x) y |ψ₀(x)|²
x = np.linspace(-3*a, 3*a, 2000)         # malla de x para dibujar (no hace falta tan fina)
fig2, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(x, psi0(x), 'b', lw=2)        # la función de onda ψ₀(x)
# axvspan(x1, x2, ...) pinta una banda vertical entre x1 y x2 → marca la región del pozo.
ax[0].axvspan(-a, a, color='gold', alpha=0.18, label='región del pozo')
ax[0].axhline(0, color='k', lw=0.7)      # línea horizontal en y=0 (eje), 'k'=negro
ax[0].set(xlabel='x [u.r.]', ylabel=r'$\psi_0(x)$',
          title=f'Función de onda del estado base (par, $E_B={E_B0:.3f}$)')
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(x, psi0(x)**2, 'purple', lw=2)               # densidad de probabilidad |ψ|²
# fill_between rellena el área entre la curva y el eje y=0 (efecto sombreado).
ax[1].fill_between(x, psi0(x)**2, color='purple', alpha=0.25)
ax[1].axvspan(-a, a, color='gold', alpha=0.18, label='región del pozo')
ax[1].set(xlabel='x [u.r.]', ylabel=r'$|\psi_0(x)|^2$',
          title='Densidad de probabilidad')
ax[1].legend(); ax[1].grid(alpha=0.3)

fig2.suptitle('Pozo cuadrado finito - estado fundamental', fontsize=13)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 6 - Probabilidad fuera del pozo (efecto túnel)
# ══════════════════════════════════════════════════════════════════════════════
# Queremos P_ext = probabilidad de hallar la partícula FUERA del pozo (|x|>a).
# Hay dos formas: (1) la fórmula exacta (analítica) y (2) integrar numéricamente.
# Que coincidan es una buena comprobación de que todo está bien.

# (1) Analítico: integral de la cola exterior, resuelta a mano:
#     P_ext = 2 ∫_{a}^{∞} B² e^{-2κx} dx = B² e^{-2κa}/κ = A² cos²(ka)/κ
P_ext_analitico = A**2 * np.cos(k*a)**2 / kap

# (2) Numérico: integrar |ψ|² sólo en las colas. OJO: hay que integrar las DOS
# colas (x < -a y x > a) por SEPARADO. Si juntáramos los puntos |x|>a en un solo
# arreglo, np.trapezoid sumaría un trapecio "fantasma" cruzando el hueco [-a, a]
# (uniría el último punto de la cola izquierda con el primero de la derecha) y
# sobreestimaría P_ext. Por eso definimos dos máscaras booleanas y sumamos aparte.
izq = x_norm <= -a       # arreglo True/False: puntos de la cola izquierda
der = x_norm >=  a       # arreglo True/False: puntos de la cola derecha
# arreglo[mascara] selecciona SÓLO los elementos donde la máscara es True.
P_ext_num = (np.trapezoid((psi0(x_norm)**2)[izq], x_norm[izq]) +
             np.trapezoid((psi0(x_norm)**2)[der], x_norm[der]))

print("\n" + "═" * 60)
print("ÍTEM 6 - Probabilidad de hallar la partícula FUERA del pozo")
print("═" * 60)
print(f"  P_ext (analítico) = {P_ext_analitico:.6f}  ({100*P_ext_analitico:.3f} %)")
print(f"  P_ext (numérico)  = {P_ext_num:.6f}  ({100*P_ext_num:.3f} %)")
print("""
  Interpretación física de |ψ₀(x)|²:
  • La partícula es MÁS probable en el centro del pozo (x = 0), donde cos²(kx)
    es máximo: el estado base concentra la densidad en el fondo del potencial.
  • Clásicamente la partícula NO podría estar en |x| > a, pues allí su energía
    cinética sería negativa (E < V = 0).  Cuánticamente la función de onda
    penetra la barrera con una cola exponencial e^{-κ|x|}: es el EFECTO TÚNEL.
  • Esa cola da una probabilidad pequeña pero NO nula de hallarla fuera del
    pozo (la penetración decae con escala 1/κ = 1/√E_B0; cuanto más ligado el
    estado, menos se asoma).
""")

# plt.show() abre las ventanas con TODAS las figuras creadas. Debe ir al final;
# el programa se queda aquí hasta que cierras las ventanas.
plt.show()
