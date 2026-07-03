"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR — FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  —  FCO4101                            ║
║                    PARCIAL III  ·  SIMULACRO  B  —  SOLUCIÓN                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Estados ligados del deuterón por el MÉTODO DE DISPARO (shooting)
  ─────────────────────────────────────────────────────────────────────────

  Pozo cuadrado:   V(x) = -V₀  si |x| < R ;   V(x) = 0  si |x| ≥ R
  con V₀ = 16 MeV, R = 10 fm.

  TISE:  ψ'' = -(2μ/ħ²)(E - V) ψ ,  escala nuclear  2μ/ħ² = 0.4829 MeV⁻¹fm⁻².

  DIFERENCIA CON EL SIMULACRO A: aquí NO usamos una fórmula trascendente ya
  deducida. Resolvemos directamente la ecuación diferencial de Schrödinger
  integrándola paso a paso (método de Runge-Kutta) y buscamos las energías E
  que producen una función de onda físicamente válida (que decae en ±∞).

  Para integrar numéricamente, una EDO de 2º orden (ψ'') se reescribe como un
  SISTEMA de dos EDOs de 1er orden, usando el vector y = [ψ, ψ']:
        y[0]' = ψ'  = y[1]
        y[1]' = ψ'' = -(2μ/ħ²)(E - V(x)) y[0]

  ALGORITMO (disparo + derivada logarítmica):
    1. Integrar ψ_L de x_min → x_match (adelante) arrancando con una cola e^{+κx}.
    2. Integrar ψ_R de x_max → x_match (atrás)  arrancando con una cola e^{-κx}.
    3. En el punto de empalme x_match comparamos las dos soluciones con
       f(E) = (ψ'_L/ψ_L − ψ'_R/ψ_R)/(ψ'_L/ψ_L + ψ'_R/ψ_R).  f(E)=0 ⇔ autovalor
       (las dos mitades empalman suavemente → ψ es una solución global válida).
    4. Bisección sobre f(E) para afinar la energía.
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Importar librerías ────────────────────────────────────────────────────────
import numpy as np               # arreglos y matemáticas (np.sqrt, np.array, ...)
import matplotlib.pyplot as plt  # gráficas

# ─── Constantes físicas y numéricas ──────────────────────────────────────────
ESCALA = 0.4829   # 2μ/ħ² = m_N/(ħc)²  [MeV⁻¹·fm⁻²]   (m_Nc²=940, ħc=197.33)
V_0    = 16.0     # [MeV] profundidad del pozo
R_pozo = 10.0     # [fm]  semiancho del pozo

EPS    = 1.0e-3   # amplitud inicial de la cola exponencial (su valor es arbitrario:
                  # la ecuación es lineal, así que sólo importa la FORMA, no la escala)
TOL_E  = 1.0e-7   # tolerancia de la bisección en energía / mismatch

# Energía de prueba y ventana de búsqueda (según enunciado)
E_prueba = -17.0
E_max0   =  1.1 * E_prueba      # = -18.7  (más profundo)
E_min0   =  E_prueba / 1.1      # = -15.45 (menos profundo)


# ─── Potencial ────────────────────────────────────────────────────────────────
def V(x):
    """Pozo cuadrado: −V₀ dentro, 0 fuera."""
    # 'A if condición else B' es un if compacto: devuelve A si se cumple, si no B.
    return -V_0 if abs(x) < R_pozo else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 1 — Sistema de EDOs y paso de Runge-Kutta 4
# ══════════════════════════════════════════════════════════════════════════════
def rhs(x, y, E):
    """Lado derecho ('right-hand side') del sistema y' = f(x,y).
    Recibe el estado y = [ψ, ψ'] y devuelve su derivada [ψ', ψ'']."""
    # np.array([...]) crea un arreglo (vector) de NumPy. Devolvemos [ψ', ψ''],
    # donde ψ'' = -(2μ/ħ²)(E-V(x))ψ sale de la ecuación de Schrödinger.
    return np.array([y[1], -ESCALA * (E - V(x)) * y[0]])

def rk4_paso(x, y, h, E):
    """Avanza UN paso de tamaño h con Runge-Kutta de 4º orden (RK4).
    RK4 estima la pendiente en 4 puntos (inicio, dos veces el medio, y el final)
    y los combina con un promedio ponderado: es mucho más preciso que avanzar con
    una sola pendiente (método de Euler). k1..k4 son esas cuatro pendientes."""
    k1 = rhs(x,           y,              E)   # pendiente al inicio del paso
    k2 = rhs(x + 0.5*h,   y + 0.5*h*k1,   E)   # pendiente en el medio (usando k1)
    k3 = rhs(x + 0.5*h,   y + 0.5*h*k2,   E)   # pendiente en el medio (usando k2)
    k4 = rhs(x + h,       y + h*k3,       E)   # pendiente al final (usando k3)
    # Nuevo estado = estado + promedio ponderado de las pendientes × paso.
    # (los puntos medios pesan el doble: por eso 2*k2 y 2*k3).
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 2 — Función de mismatch
# ══════════════════════════════════════════════════════════════════════════════
# Continuidad de la derivada logarítmica ⇔ conservación de la corriente:
# ---------------------------------------------------------------------
# La corriente de probabilidad es J = (ħ/m)·Im(ψ* ψ').  Para un estado LIGADO la
# función de onda puede tomarse REAL, de modo que J ≡ 0 en todo punto.  Empalmar
# ψ_L con ψ_R exigiendo que ψ'/ψ sea continua en x_match garantiza que, fijada la
# continuidad de ψ, también ψ' sea continua (no hay "quiebre").  Si ψ' saltara,
# ψ'' tendría una delta de Dirac → una fuente/sumidero de corriente en x_match y
# J dejaría de conservarse.  Por tanto, continuidad de ψ'/ψ ⇒ ψ' continua ⇒ J
# continua (y nula) a través del punto de unión: es la condición de autovalor.
#
# ¿Por qué la derivada logarítmica ψ'/ψ y no ψ' a secas? Porque ψ_L y ψ_R se
# integran con amplitudes arbitrarias (distintas); el cociente ψ'/ψ NO depende de
# la amplitud, así que se pueden comparar aunque tengan distinta escala.

# Parámetros de la malla usada por mismatch():
N_MATCH = 1001     # nº de puntos de la malla espacial
X_MAX   = 20.0     # [fm] el dominio debe exceder R para capturar la cola e^{-κ|x|}

def _integrar(E, N, x_max, eps=EPS):
    """Integra ψ_L (de izquierda a derecha) y ψ_R (de derecha a izquierda) hasta
    el centro x≈0.  Devuelve la malla x, los estados yL, yR y el índice de empalme.
    (El '_' inicial del nombre es una convención: 'función auxiliar interna'.)"""
    x  = np.linspace(-x_max, x_max, N)  # malla de N puntos de x, de -x_max a +x_max
    h  = x[1] - x[0]                    # paso espacial (distancia entre dos puntos)
    kappa = np.sqrt(-ESCALA * E)        # κ del decaimiento asintótico (E<0 ⇒ -E>0)
    im = N // 2                         # '//' = división entera. Índice del punto medio ≈ 0.

    # ψ_L: arranca en el extremo izquierdo con una cola e^{+κx} (que crece hacia
    # el pozo). yL es un arreglo de N filas y 2 columnas: cada fila es [ψ, ψ'].
    yL = np.zeros((N, 2))               # np.zeros((N,2)) = matriz N×2 llena de ceros
    yL[0] = [eps, kappa * eps]          # condición inicial: ψ=eps, ψ'=κ·eps (⇒ ψ∝e^{+κx})
    for i in range(im):                 # integrar hacia adelante hasta el centro
        yL[i+1] = rk4_paso(x[i], yL[i], h, E)   # un paso RK4 (h>0 = hacia la derecha)

    # ψ_R: arranca en el extremo derecho con una cola e^{-κx} y se integra HACIA
    # ATRÁS (paso negativo -h). Visto desde la derecha, e^{-κx} crece hacia el pozo.
    yR = np.zeros((N, 2))
    yR[-1] = [eps, -kappa * eps]        # yR[-1] = última fila (extremo derecho)
    for i in range(N - 1, im, -1):      # range(inicio, fin, paso): cuenta hacia atrás
        yR[i-1] = rk4_paso(x[i], yR[i], -h, E)  # paso -h = hacia la izquierda

    return x, yL, yR, im

def mismatch(E):
    """f(E) = (ψ'_L/ψ_L − ψ'_R/ψ_R)/(ψ'_L/ψ_L + ψ'_R/ψ_R) en x_match.
    Vale 0 cuando las derivadas logarítmicas de ambas mitades coinciden."""
    _, yL, yR, im = _integrar(E, N_MATCH, X_MAX)   # el '_' descarta el valor x (no lo usamos)
    dL = yL[im, 1] / yL[im, 0]    # ψ'_L/ψ_L en el empalme. yL[im,1]=ψ', yL[im,0]=ψ
    dR = yR[im, 1] / yR[im, 0]    # ψ'_R/ψ_R en el empalme
    return (dL - dR) / (dL + dR)  # diferencia relativa (normalizada entre -1 y 1)

def wronskiano(E):
    """W(E) = ψ_L ψ'_R − ψ'_L ψ_R en x_match (normalizado).  También vale 0 en los
    autovalores, pero —a diferencia de mismatch()— es SUAVE (no tiene polos donde
    ψ se anula). Por eso lo usamos para BARRER y contar todos los estados."""
    _, yL, yR, im = _integrar(E, N_MATCH, X_MAX)
    L  = yL[im] / abs(yL[im, 0])   # dividir por la amplitud quita la escala arbitraria
    Rg = yR[im] / abs(yR[im, 0])
    return L[0]*Rg[1] - L[1]*Rg[0] # ψ_L·ψ'_R − ψ'_L·ψ_R


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 3 — Bisección sobre mismatch(E)  →  estado fundamental
# ══════════════════════════════════════════════════════════════════════════════
def biseccion(f, a, b, tol=TOL_E, Nmax=200):
    """Raíz de f en [a, b] por bisección (parte el intervalo a la mitad
    repetidamente). Requiere que f(a) y f(b) tengan signos opuestos."""
    fa = f(a)
    for _ in range(Nmax):
        c, fc = 0.5*(a+b), f(0.5*(a+b))     # punto medio y su valor de f
        if abs(fc) < tol or 0.5*abs(b-a) < tol:   # ¿ya convergió?
            return c
        if fa*fc < 0.0:        # raíz entre a y c
            b = c
        else:                  # raíz entre c y b
            a, fa = c, fc
    return 0.5*(a+b)

print("═"*64)
print("ÍTEM 3 — Búsqueda del estado fundamental por bisección")
print("═"*64)
# sorted((p,q)) ordena la pareja de menor a mayor → a=-18.7 (menor), b=-15.45 (mayor).
a, b = sorted((E_min0, E_max0))     # [-18.7, -15.45]
fa, fb = mismatch(a), mismatch(b)
print(f"Ventana inicial: [{a:.4f}, {b:.4f}] MeV   f(a)={fa:.3e}  f(b)={fb:.3e}")
E_fund = biseccion(mismatch, a, b)  # afinamos la energía donde mismatch=0
print(f"\n>>> Estado fundamental del deuterón:  E = {E_fund:.6f} MeV"
      f"   (E_enlace = {-E_fund:.4f} MeV)")

# Comprobación independiente: barremos toda la ventana física (−V₀, 0) con el
# Wronskiano para CONTAR cuántos estados ligados hay y verificar el fundamental.
Es = np.linspace(-V_0 + 1e-3, -1e-3, 3000)   # malla de energías a probar
w_prev, niveles = wronskiano(Es[0]), []      # w_prev = Wronskiano en el 1er punto
for E in Es[1:]:                             # recorremos el resto de energías
    w = wronskiano(E)
    if w_prev * w < 0.0:                     # cambió de signo → hay un autovalor entre medio
        niveles.append(biseccion(wronskiano, E - (Es[1]-Es[0]), E))   # lo afinamos
    w_prev = w                               # guardamos para la siguiente comparación
niveles.sort()                               # ordena la lista de menor a mayor (in situ)
print(f"\nComprobación (barrido del Wronskiano): {len(niveles)} estados ligados.")
print(f"  · El más ligado (n=0) = {min(niveles):.4f} MeV  ← coincide con la bisección")
print(f"  · El menos ligado     = {max(niveles):.4f} MeV")


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 4 — Re-integración de alta resolución y gráfica de ψ(x)
# ══════════════════════════════════════════════════════════════════════════════
# Ya tenemos la energía; ahora reconstruimos la función de onda completa con más
# puntos (N_PLOT) para graficarla bien.
N_PLOT = 1501
x, yL, yR, im = _integrar(E_fund, N_PLOT, 20.0)

# ψ_L y ψ_R se integraron con amplitudes distintas. Para unirlas en una sola curva
# continua, reescalamos la mitad izquierda con un 'factor' que iguale ambas en el
# punto de empalme: factor = ψ_R(empalme) / ψ_L(empalme).
factor = yR[im, 0] / yL[im, 0]
psi = np.empty(N_PLOT)                # np.empty = arreglo sin inicializar (lo llenamos ya)
psi[:im+1] = factor * yL[:im+1, 0]    # tramo izquierdo (reescalado). [:im+1] = del 0 al im
psi[im+1:] = yR[im+1:, 0]             # tramo derecho. [im+1:] = del im+1 al final

# Contar nodos = cuántas veces ψ cruza el cero (cambia de signo). El estado base
# no tiene nodos; el n-ésimo tiene n nodos.
mask = np.abs(psi) > 1e-3 * np.max(np.abs(psi))   # ignorar las colas (ψ≈0) para no contar ruido
# np.sign da -1/0/+1 según el signo; np.diff resta elementos consecutivos; un cambio
# de signo produce un valor ≠ 0. np.sum cuenta cuántos cambios hubo.
nodos = np.sum(np.diff(np.sign(psi[mask])) != 0)
# Si ψ tiene el mismo signo en ambos extremos → simétrica (par); si no → impar.
simetria = "SIMÉTRICA (par)" if psi[0]*psi[-1] > 0 else "ANTISIMÉTRICA (impar)"

print("\n" + "═"*64)
print("ÍTEM 4 — Función de onda del estado fundamental")
print("═"*64)
print(f"  Paridad: {simetria}   ·   nodos = {nodos}")
print("  El estado base es PAR (∝cos kx, sin nodos): es la configuración de")
print("  mínima energía cinética, por lo que siempre es el más ligado.")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))   # figura con 2 paneles
ax[0].plot(x, psi, 'b', lw=2)                      # la función de onda ψ(x)
# axvspan pinta la banda del pozo |x|<R. axvline/axhline = líneas vertical/horizontal.
ax[0].axvspan(-R_pozo, R_pozo, color='gold', alpha=0.18, label=f'pozo |x|<{R_pozo} fm')
ax[0].axvline(0, color='gray', ls='--', alpha=0.6)
ax[0].axhline(0, color='k', lw=0.7)
ax[0].set(xlabel='x [fm]', ylabel=r'$\psi(x)$',
          title=f'Función de onda  (E = {E_fund:.4f} MeV)')
ax[0].legend(); ax[0].grid(alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 5 — Densidad de probabilidad y probabilidades dentro/fuera
# ══════════════════════════════════════════════════════════════════════════════
dens = psi**2                           # densidad de probabilidad |ψ|² (psi al cuadrado)
norm = np.trapezoid(dens, x)            # ∫|ψ|² dx por la regla del trapecio (integración numérica)
dens /= norm                            # 'dens /= norm' equivale a 'dens = dens/norm': normaliza a 1
psi  /= np.sqrt(norm)                   # ψ también se reescala (su cuadrado es dens)

# Probabilidad dentro y fuera del pozo. Incluimos x=±R en AMBOS lados: cada región
# aporta el intervalo de su lado, sin huecos ni solapes.
dentro = np.abs(x) <= R_pozo            # máscara True/False: |x| ≤ R
P_in  = np.trapezoid(dens[dentro], x[dentro])   # integra sólo donde la máscara es True
# Las dos colas exteriores se integran por SEPARADO: si se unieran en un solo
# arreglo, trapezoid sumaría un trapecio espurio cruzando el pozo (P_out inflada).
izq, der = x <= -R_pozo, x >= R_pozo
P_out = np.trapezoid(dens[izq], x[izq]) + np.trapezoid(dens[der], x[der])

print("\n" + "═"*64)
print("ÍTEM 5 — Densidad de probabilidad")
print("═"*64)
print(f"  P(|x| <  R) dentro del pozo = {P_in:.5f}  ({100*P_in:.2f} %)")
print(f"  P(|x| ≥  R) fuera  del pozo = {P_out:.5f}  ({100*P_out:.2f} %)")
print(f"  (suma = {P_in+P_out:.5f})")
print("  El nucleón está confinado casi por completo en |x|<R: el estado base")
print("  está muy ligado (E_enlace≈V₀), su κ=√(2μE_B/ħ²) es grande y la cola")
print("  exterior e^{-κ|x|} decae en ~1/κ fm → refleja el CORTO ALCANCE (R) de")
print("  la fuerza nuclear: fuera del pozo la interacción se anula.")

ax[1].plot(x, dens, 'steelblue', lw=2)                      # |ψ|² normalizada
ax[1].fill_between(x, dens, color='steelblue', alpha=0.3)   # sombrea el área bajo la curva
ax[1].axvspan(-R_pozo, R_pozo, color='gold', alpha=0.18)
ax[1].set(xlabel='x [fm]', ylabel=r'$|\psi(x)|^2$',
          title=f'Densidad normalizada  (P_dentro = {100*P_in:.1f}%)')
ax[1].grid(alpha=0.3)
fig.suptitle(f'Deuterón — pozo cuadrado  $V_0={V_0}$ MeV, $R={R_pozo}$ fm  '
             f'→  $E={E_fund:.3f}$ MeV', fontsize=13)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 6 — Variación de V₀ y condición de umbral
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*64)
print("ÍTEM 6 — Pozo menos profundo (V₀ = 10 MeV) y umbral de ligadura")
print("═"*64)

# Cambiamos V_0 a 10. Como V_0 es una variable GLOBAL, la función V(x) (definida
# arriba) usará automáticamente este nuevo valor en lo que sigue.
V_0 = 10.0

# (a) Intentar la MISMA ventana del enunciado (alrededor de E = -17 MeV):
a, b = sorted((E_min0, E_max0))
# 'try/except' ejecuta el bloque y, si ocurre un error numérico, lo captura en
# 'except' en vez de detener el programa.
try:
    fa, fb = mismatch(a), mismatch(b)
    print(f"  Ventana [-18.7, -15.45] MeV:  f(a)·f(b) = {fa*fb:+.3e}")
    if fa*fb > 0:           # mismo signo en ambos extremos → no hay raíz que biseccionar
        print("  → NO hay cambio de signo: la bisección NO converge.")
except Exception as e:
    print(f"  → fallo numérico: {e}")
print("  Motivo físico: con V₀=10 el fondo del pozo es -10 MeV, así que NINGÚN")
print("  estado puede tener E<-10 MeV.  La ventana centrada en -17 MeV queda")
print("  POR DEBAJO del pozo (zona prohibida): no existe raíz ahí.")

# (b) Búsqueda correcta en (−V₀, 0): demostramos que el estado ligado SIGUE existiendo.
Es = np.linspace(-V_0 + 1e-3, -1e-3, 3000)
w_prev, niveles10 = wronskiano(Es[0]), []
for E in Es[1:]:
    w = wronskiano(E)
    if w_prev * w < 0.0:
        niveles10.append(biseccion(wronskiano, E - (Es[1]-Es[0]), E))
    w_prev = w
niveles10.sort()
print(f"\n  Búsqueda correcta en (-10, 0):  {len(niveles10)} estados ligados.")
print(f"  Estado fundamental con V₀=10:  E = {min(niveles10):.4f} MeV")
print("  ⇒ El pozo SÍ liga: en 1D un pozo simétrico SIEMPRE tiene ≥1 estado par.")

# (c) Umbral para EXACTAMENTE un estado ligado (R = 10 fm).
#     El 2º estado (1er impar) aparece cuando z₀ = R√(2μV₀/ħ²) = π/2.
#     ⇒ V₀,umbral = (π/2)² / (ESCALA·R²).
V0_umbral = (np.pi/2)**2 / (ESCALA * R_pozo**2)
print(f"\n  Umbral (z₀ = R√(2μV₀/ħ²) = π/2, E_B→0 del 2º nivel):")
print(f"     V₀,min = (π/2)² / (2μ/ħ² · R²) = {V0_umbral:.4f} MeV")
print(f"  Para 0 < V₀ < {V0_umbral:.3f} MeV el pozo (R=10 fm) liga EXACTAMENTE un")
print("  estado (el par fundamental).  Por encima aparecen niveles adicionales.")

V_0 = 16.0   # restaurar el valor original (buena práctica: dejar el estado como estaba)

plt.show()   # abre todas las figuras (al final del programa)
