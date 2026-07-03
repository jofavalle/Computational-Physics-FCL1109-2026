"""
Librería Física Computacional Común (LFCC) - «fcl1109.py»
============================================================

Librería de métodos numéricos desarrollada para el curso FCL1109.
Contiene herramientas para:

  1. Derivación numérica (adelante, central, n-ésima derivada)
  2. Integración numérica (trapecio, Simpson, Monte Carlo, cambio de variable)
  3. Resolución de EDOs (Euler, Runge-Kutta 4)
  4. Ajuste de datos (mínimos cuadrados, chi cuadrada)
  5. Búsqueda de raíces (Newton-Raphson)
  6. Análisis de Fourier (DFT, IDFT, FFT, IFFT)
"""

import numpy as np

# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Paso por defecto para la derivada numérica, usado internamente por derivada_nesima.
# Se define aquí para no tener que pasarlo como argumento en cada nivel de recursión.
_h_default = 1e-5


# ============================================================================
# 1. DERIVACIÓN NUMÉRICA
# ============================================================================

def derivada_adelante(f, x, h):
    """
    Calcula la derivada de f en el punto x usando diferencias finitas hacia adelante.

    Esta es la fórmula más simple de derivación numérica. Aproxima la pendiente
    de la recta tangente usando el punto actual y un punto ligeramente adelante.

    Fórmula:  f'(x) ≈ [f(x + h) - f(x)] / h

    Tiene un error de orden O(h), es decir, si reducimos h a la mitad,
    el error se reduce aproximadamente a la mitad también.

    Parámetros:
        f : función    - La función a derivar. Debe aceptar un número y devolver un número.
        x : float      - El punto donde queremos evaluar la derivada.
        h : float      - El tamaño del paso. Valores típicos: 1e-5 a 1e-8.

    Retorna:
        float - Aproximación numérica de f'(x).

    Ejemplo:
        >>> derivada_adelante(np.sin, 0.0, 1e-5)
        0.9999999999833334  # ≈ cos(0) = 1
    """
    return (f(x + h) - f(x)) / h


def derivada_central(f, x, h):
    """
    Calcula la derivada de f en el punto x usando diferencias finitas centrales.

    A diferencia de la derivada hacia adelante, esta fórmula evalúa la función
    a ambos lados del punto x, lo que produce una mejor cancelación de errores.

    Fórmula:  f'(x) ≈ [f(x + h/2) - f(x - h/2)] / h

    Tiene un error de orden O(h²), es decir, es mucho más precisa que la
    derivada hacia adelante para el mismo valor de h.

    Parámetros:
        f : función    - La función a derivar.
        x : float      - El punto donde queremos evaluar la derivada.
        h : float      - El tamaño del paso. Valores típicos: 1e-5 a 1e-8.

    Retorna:
        float - Aproximación numérica de f'(x).

    Ejemplo:
        >>> derivada_central(lambda x: x**3, 2.0, 1e-5)
        12.000000000000  # ≈ 3·(2)² = 12
    """
    return (f(x + h/2) - f(x - h/2)) / h


def derivada_enesima(f, x, n):
    """
    Calcula la n-ésima derivada de f en el punto x de forma recursiva.

    Utiliza la derivada central internamente. Para calcular la segunda derivada,
    primero construye una nueva función que representa la primera derivada,
    y luego le aplica la derivada central otra vez. Y así sucesivamente.

    Usa el paso global _h_default (= 1e-5) definido al inicio del archivo.

    Parámetros:
        f : función    - La función a derivar.
        x : float      - El punto donde evaluar la derivada.
        n : int        - El orden de la derivada (0 = la función misma, 1 = primera, 2 = segunda, etc.)

    Retorna:
        float - Aproximación numérica de f⁽ⁿ⁾(x).

    Nota:
        La precisión disminuye conforme n aumenta, porque los errores numéricos
        se acumulan en cada nivel de recursión.

    Ejemplo:
        >>> derivada_enesima(lambda x: x**3, 2.0, 1)   # Primera derivada: 3x² = 12
        12.0
        >>> derivada_enesima(lambda x: x**3, 2.0, 2)   # Segunda derivada: 6x = 12
        12.0
    """
    if n == 0:
        # Caso base: la «derivada de orden 0» es simplemente evaluar la función
        return f(x)
    else:
        # Caso recursivo: construimos g(y) = f'(y) usando derivada central,
        # y luego calculamos la (n-1)-ésima derivada de g
        return derivada_enesima(lambda y: derivada_central(f, y, _h_default), x, n - 1)


# ============================================================================
# 2. INTEGRACIÓN NUMÉRICA
# ============================================================================

def trapecio(f, a, b, n):
    """
    Calcula la integral definida de f(x) en [a, b] usando el método del trapecio compuesto.

    La idea es dividir el intervalo [a, b] en n subintervalos iguales y aproximar
    el área bajo la curva como la suma de áreas de trapecios. Cada trapecio se
    forma conectando dos puntos consecutivos de la función con una línea recta.

    Fórmula:
        ∫ₐᵇ f(x)dx ≈ h·[½·f(x₀) + f(x₁) + f(x₂) + ... + f(xₙ₋₁) + ½·f(xₙ)]

    donde h = (b - a)/n es el ancho de cada subintervalo.

    Tiene un error de orden O(h²), proporcional a la segunda derivada de f.

    Parámetros:
        f : función    - La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      - Límite inferior de integración.
        b : float      - Límite superior de integración.
        n : int        - Número de subintervalos (más grande = más preciso).

    Retorna:
        float - Aproximación numérica de ∫ₐᵇ f(x)dx.

    Ejemplo:
        >>> trapecio(lambda x: x**2, 0, 1, 1000)
        0.33333350   # ≈ 1/3
    """
    # Crear n+1 puntos equiespaciados en [a, b]
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n           # ancho de cada subintervalo
    y = f(x)                  # evaluar la función en todos los puntos

    # Aplicar la fórmula del trapecio compuesto:
    # El primer y último punto llevan factor 1/2, los intermedios factor 1
    I = h * (0.5 * y[0] + np.sum(y[1:n]) + 0.5 * y[n])
    return I


def simpson(f, a, b, n):
    """
    Calcula la integral definida de f(x) en [a, b] usando el método de Simpson compuesto.

    Simpson aproxima la función con parábolas (polinomios de grado 2) en cada par
    de subintervalos consecutivos, lo que da mayor precisión que el trapecio.

    Requiere un número par de subintervalos. Si se pasa un n impar, la función
    lo incrementa automáticamente en 1 para hacerlo par.

    Fórmula:
        ∫ₐᵇ f(x)dx ≈ (h/3)·[f(x₀) + 4·f(x₁) + 2·f(x₂) + 4·f(x₃) + ... + f(xₙ)]

    donde h = (b - a)/n. Los coeficientes alternan: 1, 4, 2, 4, 2, ..., 4, 1.

    Tiene un error de orden O(h⁴), mucho más preciso que el trapecio.

    Parámetros:
        f : función    - La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      - Límite inferior de integración.
        b : float      - Límite superior de integración.
        n : int        - Número de subintervalos (se ajusta a par si es impar).

    Retorna:
        float - Aproximación numérica de ∫ₐᵇ f(x)dx.

    Ejemplo:
        >>> simpson(lambda x: x**2, 0, 1, 1000)
        0.33333333333333   # Exacto para polinomios de grado ≤ 3
    """
    # Simpson requiere n par; si es impar, lo corregimos
    if n % 2 == 1:
        n += 1

    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    y = f(x)

    # Fórmula de Simpson compuesto:
    # y[1:n:2] son los puntos en posiciones impares (coeficiente 4)
    # y[2:n-1:2] son los puntos en posiciones pares intermedias (coeficiente 2)
    I = h / 3 * (y[0] + y[n] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]))
    return I


def montecarlo(f, a, b, N):
    """
    Calcula la integral definida de f(x) en [a, b] usando el método de Monte Carlo.

    Genera N puntos aleatorios uniformes en [a, b], evalúa la función en esos puntos,
    y estima la integral como el promedio de los valores multiplicado por el ancho
    del intervalo. Es un método probabilístico: cada ejecución da un resultado
    ligeramente distinto.

    Fórmula:
        ∫ₐᵇ f(x)dx ≈ (b - a) · (1/N) · Σ f(xᵢ)

    Su error decrece como O(1/√N), independientemente de la dimensión del problema,
    lo que lo hace especialmente útil para integrales multidimensionales.

    Parámetros:
        f : función    - La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      - Límite inferior de integración.
        b : float      - Límite superior de integración.
        N : int        - Número de puntos aleatorios (más grande = más preciso pero más lento).

    Retorna:
        float - Aproximación numérica de ∫ₐᵇ f(x)dx.

    Ejemplo:
        >>> np.random.seed(42)
        >>> montecarlo(lambda x: x**2, 0, 1, 100000)
        0.3326   # ≈ 1/3 (varía en cada ejecución)
    """
    # Generar N puntos aleatorios uniformemente distribuidos en [a, b]
    x = np.random.uniform(a, b, N)
    # La integral se estima como (longitud del intervalo) × (promedio de f)
    I = (b - a) * np.mean(f(x))
    return I


# --- Cambio de variable para integrales impropias (0 a ∞) ---
#
# Para calcular ∫₀^∞ f(x)dx numéricamente, hacemos el cambio de variable:
#     x = z / (1 - z)        con z ∈ [0, 1]
#     dx = 1 / (1 - z)²  dz   (este es el jacobiano)
#
# Así transformamos la integral impropia en una integral en [0, 1]:
#     ∫₀^∞ f(x)dx = ∫₀¹ f(z/(1-z)) · 1/(1-z)² dz
#
# En la práctica, integramos en [ε, 1-ε] para evitar la singularidad en z=1.

def cambio_variable(z):
    """
    Realiza el cambio de variable x = z/(1-z) para transformar [0,1] → [0,∞).

    Esta función es parte del trío de funciones para integrales impropias.
    Dado un valor z ∈ (0, 1), devuelve el correspondiente x ∈ (0, ∞).

    Parámetros:
        z : float - Valor en el intervalo (0, 1).

    Retorna:
        float - El valor x = z/(1-z) en el intervalo (0, ∞).

    Ejemplo:
        >>> cambio_variable(0.5)
        1.0       # z=0.5 corresponde a x=1
        >>> cambio_variable(0.75)
        3.0       # z=0.75 corresponde a x=3
    """
    return z / (1.0 - z)


def jacobiano(z):
    """
    Calcula el jacobiano dx/dz = 1/(1-z)² del cambio de variable x = z/(1-z).

    Al hacer un cambio de variable en una integral, necesitamos multiplicar
    por |dx/dz| para compensar la «deformación» del intervalo.

    Parámetros:
        z : float - Valor en el intervalo (0, 1).

    Retorna:
        float - El jacobiano 1/(1-z)².

    Ejemplo:
        >>> jacobiano(0.5)
        4.0
    """
    return 1.0 / (1.0 - z)**2


def integrando_transformado(f, z):
    """
    Transforma el integrando f(x) para poder integrarlo en [0, 1] en vez de [0, ∞).

    Combina el cambio de variable y el jacobiano: devuelve f(z/(1-z)) · 1/(1-z)².
    El resultado de esta función se pasa directamente a simpson() o trapecio()
    con límites cercanos a [0, 1], por ejemplo (1e-6, 1 - 1e-6).

    Parámetros:
        f : función    - La función original que queremos integrar de 0 a ∞.
        z : float      - Punto en (0, 1) donde evaluar el integrando transformado.

    Retorna:
        float - Valor del integrando transformado en z.

    Ejemplo:
        Para calcular ∫₀^∞ e^(-x) dx = 1:
        >>> eps = 1e-6
        >>> simpson(lambda z: integrando_transformado(np.exp(-x), z), eps, 1-eps, 1000)
        0.999999   # ≈ 1
    """
    x = cambio_variable(z)
    return f(x) * jacobiano(z)


def integral_impropia(f, a, b, n=1000, metodo=None):
    """
    Calcula integrales impropias (con límites infinitos) de forma automática.

    Detecta si alguno de los límites es ±∞ y aplica el cambio de variable
    adecuado para transformar la integral en una integral sobre [0, 1], que
    luego se evalúa con el método numérico indicado (Simpson por defecto).

    Casos soportados:
        1. ∫ₐ^∞  f(x)dx   - Solo el límite superior es infinito.
           Cambio: x = a + z/(1-z),  dx = 1/(1-z)² dz,  z ∈ [0, 1]

        2. ∫₋∞^b  f(x)dx  - Solo el límite inferior es infinito.
           Cambio: x = b - z/(1-z),  dx = 1/(1-z)² dz,  z ∈ [0, 1]
           (se refleja el eje, recorriendo de b hacia -∞)

        3. ∫₋∞^∞  f(x)dx  - Ambos límites son infinitos.
           Se parte en dos: ∫₋∞^0 f(x)dx + ∫₀^∞ f(x)dx
           y se aplican los casos 2 y 1 respectivamente.

    Parámetros:
        f      : función - La función a integrar. Debe aceptar float/array.
        a      : float   - Límite inferior. Usar -np.inf para -∞.
        b      : float   - Límite superior. Usar np.inf para +∞.
        n      : int     - Número de subintervalos para el método numérico (default: 1000).
        metodo : función - Método de integración a usar (default: simpson).
                           Debe tener firma metodo(f, a, b, n).

    Retorna:
        float - Aproximación numérica de la integral.

    Ejemplos:
        >>> integral_impropia(lambda x: np.exp(-x), 0, np.inf)
        1.0                    # ∫₀^∞ e^(-x) dx = 1

        >>> integral_impropia(lambda x: np.exp(x), -np.inf, 0)
        1.0                    # ∫₋∞^0 e^(x) dx = 1

        >>> integral_impropia(lambda x: np.exp(-x**2), -np.inf, np.inf)
        1.7724538509           # ∫₋∞^∞ e^(-x²) dx = √π

        >>> integral_impropia(lambda x: 1/(1+x**2), 5, np.inf)
        0.19739555985          # ∫₅^∞ 1/(1+x²) dx = π/2 - arctan(5)
    """
    if metodo is None:
        metodo = simpson

    eps = 1e-10  # para evitar singularidades en z=0 y z=1

    a_inf = (a == -np.inf)
    b_inf = (b == np.inf)

    if a_inf and b_inf:
        # Caso 3: ∫₋∞^∞ - partir en ∫₋∞^0 + ∫₀^∞
        I1 = integral_impropia(f, -np.inf, 0, n, metodo)
        I2 = integral_impropia(f, 0, np.inf, n, metodo)
        return I1 + I2

    elif b_inf:
        # Caso 1: ∫ₐ^∞ - cambio x = a + z/(1-z)
        def g(z):
            x = a + z / (1.0 - z)
            return f(x) / (1.0 - z)**2
        return metodo(g, eps, 1 - eps, n)

    elif a_inf:
        # Caso 2: ∫₋∞^b - cambio x = b - z/(1-z)
        def g(z):
            x = b - z / (1.0 - z)
            return f(x) / (1.0 - z)**2
        return metodo(g, eps, 1 - eps, n)

    else:
        # Caso sin infinitos: integral ordinaria
        return metodo(f, a, b, n)


# ============================================================================
# 3. RESOLUCIÓN DE ECUACIONES DIFERENCIALES ORDINARIAS (EDOs)
# ============================================================================

def euler(f, t, x, h):
    """
    Realiza UN paso del método de Euler para resolver una EDO.

    El método de Euler es el integrador numérico más simple. Dada una EDO
    de la forma dx/dt = f(t, x), avanza la solución un paso temporal h usando:

        x(t + h) ≈ x(t) + h · f(t, x(t))

    Es un método de orden 1: el error por paso es O(h²) y el error acumulado es O(h).
    Para mayor precisión, usar rk4().

    Funciona tanto para EDOs escalares (x es un float) como para sistemas de EDOs
    (x es un array de NumPy), lo que permite resolver sistemas acoplados.
    La firma f(t, x) es idéntica a la de rk4(), por lo que ambos integradores
    son intercambiables.

    Nota: esta función da UN solo paso. Para integrar en un intervalo completo,
    hay que llamarla repetidamente en un bucle.

    Parámetros:
        f : función        - El lado derecho de la EDO dx/dt = f(t, x).
                             Debe tener firma f(t, x) → dx/dt.
        t : float          - El tiempo actual.
        x : float o array  - El estado actual (escalar para 1 EDO, array para sistemas).
        h : float          - El tamaño del paso temporal.

    Retorna:
        float o array - El nuevo estado x(t + h).

    Ejemplo (EDO escalar):
        Para dx/dt = -x con x(0) = 1, integrar hasta t = 1 con dt = 0.001:
        >>> x, t = 1.0, 0.0
        >>> for _ in range(1000):
        ...     x = euler(lambda t, x: -x, t, x, 0.001)
        ...     t += 0.001
        >>> print(x)
        0.3677   # ≈ e^(-1) = 0.3679

    Ejemplo (sistema de EDOs):
        Para el oscilador armónico: d/dt [x, v] = [v, -x]
        >>> estado = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
        >>> f = lambda t, s: np.array([s[1], -s[0]])
        >>> estado = euler(f, 0.0, estado, 0.01)
    """
    return x + h * f(t, x)


def rk4(f, t, x, h):
    """
    Realiza UN paso del método de Runge-Kutta de orden 4 (RK4) para resolver una EDO.

    RK4 es el método de referencia para la integración numérica de EDOs. Evalúa la
    función f en 4 puntos dentro del paso y combina esos valores para obtener
    un error por paso de orden O(h⁵), considerablemente menor que el de Euler.

    Dada una EDO de la forma dx/dt = f(t, x), el algoritmo calcula:
        k₁ = h · f(t, x)
        k₂ = h · f(t + h/2, x + k₁/2)
        k₃ = h · f(t + h/2, x + k₂/2)
        k₄ = h · f(t + h, x + k₃)
        x(t + h) = x(t) + (k₁ + 2k₂ + 2k₃ + k₄) / 6

    Funciona tanto para EDOs escalares (x es un float) como para sistemas de EDOs
    (x es un array de NumPy), lo que permite resolver sistemas acoplados.

    Parámetros:
        t : float          - El tiempo actual.
        h : float          - El tamaño del paso temporal.
        x : float o array  - El estado actual (escalar para 1 EDO, array para sistemas).
        f : función        - El lado derecho de la EDO. Debe tener firma f(t, x) → dx/dt.

    Retorna:
        float o array - El nuevo estado x(t + h).

    Ejemplo (EDO escalar):
        Para dx/dt = -x con x(0) = 1:
        >>> x, t = 1.0, 0.0
        >>> for i in range(100):
        ...     x = rk4(t, 0.01, x, lambda t, x: -x)
        ...     t += 0.01
        >>> print(x)
        0.3678794412   # ≈ e^(-1), con error ~ 10⁻¹¹

    Ejemplo (sistema de EDOs):
        Para el oscilador armónico: d/dt [x, v] = [v, -x]
        >>> estado = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
        >>> f = lambda t, s: np.array([s[1], -s[0]])
        >>> estado = rk4(0.0, 0.01, estado, f)
    """
    k1 = h * f(t, x)
    k2 = h * f(t + h/2, x + k1/2)
    k3 = h * f(t + h/2, x + k2/2)
    k4 = h * f(t + h, x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6


# ============================================================================
# 4. AJUSTE DE DATOS
# ============================================================================

def minimos_cuadrados(A, b):
    """
    Resuelve el sistema de ecuaciones normales A·c = b para obtener los coeficientes
    del ajuste por mínimos cuadrados.

    En un ajuste lineal generalizado (que puede ser un polinomio de cualquier grado),
    construimos la matriz A y el vector b a partir de los datos y las incertidumbres,
    y luego resolvemos el sistema para encontrar los coeficientes óptimos.

    Para un ajuste lineal y = a₀ + a₁x, la matriz y vector serían:
        A = [[Σ(1/σ²),    Σ(x/σ²)  ],       b = [Σ(y/σ²)  ]
             [Σ(x/σ²),   Σ(x²/σ²) ]]            [Σ(xy/σ²) ]

    Para un polinomio de grado mayor, A es de tamaño (grado+1) × (grado+1).

    Parámetros:
        A : array 2D (n×n)  - La matriz de coeficientes del sistema normal.
        b : array 1D (n)    - El vector del lado derecho.

    Retorna:
        array 1D (n) - Los coeficientes del ajuste [a₀, a₁, ..., aₙ₋₁].

    Ejemplo:
        >>> A = np.array([[20, 10], [10, 90]])
        >>> b = np.array([15, 85])
        >>> minimos_cuadrados(A, b)
        array([0.5, 0.889])  # coeficientes del ajuste
    """
    coeficientes = np.linalg.solve(A, b)
    return coeficientes


def chi_cuadrada(y_observado, y_ajustado, sigma):
    """
    Calcula el estadístico chi cuadrada (χ²) para evaluar la calidad de un ajuste.

    El χ² mide qué tan bien el modelo ajustado describe los datos observados,
    teniendo en cuenta las incertidumbres (σ) de cada punto.

    Fórmula:
        χ² = Σᵢ [(yᵢ_obs - yᵢ_ajust) / σᵢ]²

    Interpretación:
        - Si χ²/ν ≈ 1 (donde ν = N - p = grados de libertad), el ajuste es bueno.
        - Si χ²/ν >> 1, el modelo no describe bien los datos.
        - Si χ²/ν << 1, las incertidumbres están sobreestimadas.

    Parámetros:
        y_observado : array - Valores medidos experimentalmente.
        y_ajustado  : array - Valores predichos por el modelo.
        sigma       : array - Incertidumbres de cada medición.

    Retorna:
        float - El valor de χ².

    Ejemplo:
        >>> y_obs = np.array([1.1, 2.0, 2.9])
        >>> y_fit = np.array([1.0, 2.0, 3.0])
        >>> sigma = np.array([0.1, 0.1, 0.1])
        >>> chi_cuadrada(y_obs, y_fit, sigma)
        2.0
    """
    chi2 = np.sum(((y_observado - y_ajustado) / sigma) ** 2)
    return chi2


# ============================================================================
# 5. BÚSQUEDA DE RAÍCES - NEWTON-RAPHSON
# ============================================================================

def newton_raphson(f, x, dx, eps, Nmax):
    """
    Encuentra una raíz de f(x) = 0 usando el método de Newton-Raphson.

    Newton-Raphson es un método iterativo que, a partir de una estimación inicial x₀,
    mejora la aproximación usando la fórmula:

        x_{n+1} = x_n - f(x_n) / f'(x_n)

    En esta implementación, la derivada f'(x) se calcula numéricamente usando
    diferencias centrales con paso dx, así no hace falta conocer la derivada analítica.

    El método converge cuadráticamente (los dígitos correctos se duplican en cada
    iteración) si la estimación inicial está suficientemente cerca de la raíz.

    Parámetros:
        f    : función - La función cuya raíz buscamos.
        x    : float   - Estimación inicial de la raíz.
        dx   : float   - Paso para calcular la derivada numérica (típico: 1e-6).
        eps  : float   - Tolerancia: el algoritmo se detiene cuando |f(x)| ≤ eps.
        Nmax : int     - Número máximo de iteraciones (protección contra no-convergencia).

    Retorna:
        float - La raíz aproximada.

    Nota:
        Si el método no converge en Nmax iteraciones, imprime un aviso por consola
        y devuelve la última aproximación obtenida.

    Ejemplo:
        >>> newton_raphson(lambda x: x**2 - 4, x=3.0, dx=1e-6, eps=1e-10, Nmax=100)
        2.0
    """
    for it in range(Nmax):
        F = f(x)

        # Si ya estamos lo suficientemente cerca de la raíz, terminamos
        if abs(F) <= eps:
            break
        else:
            # Calcular la derivada numéricamente con diferencias centrales
            df = (f(x + dx/2) - f(x - dx/2)) / dx
            # Aplicar la fórmula de Newton: x_new = x - f(x)/f'(x)
            x += -F / df
    else:
        # Este bloque se ejecuta si el for termina sin break (no convergió)
        print("Newton-Raphson no convergió para Nmax =", Nmax)

    return x


# ============================================================================
# 6. ANÁLISIS DE FOURIER
# ============================================================================

def dft(x):
    """
    Calcula la Transformada Discreta de Fourier (DFT) de una señal real.

    Dada una señal x[n] con N puntos, calcula los coeficientes de Fourier X[k]
    para k = 0, 1, ..., N/2 (solo la mitad positiva, ya que para señales reales
    la otra mitad es el conjugado).

    Fórmula:
        X[k] = Σₙ x[n] · e^(-2πi·k·n/N)     para k = 0, 1, ..., N/2

    Este es un algoritmo directo con complejidad O(N²). Para señales grandes,
    es preferible usar fft() que tiene complejidad O(N·log N).

    Parámetros:
        x : array - La señal de entrada (N muestras, valores reales).

    Retorna:
        array complejo - Los N/2 + 1 coeficientes de Fourier X[k].

    Ejemplo:
        >>> señal = np.cos(2 * np.pi * 3 * np.arange(64) / 64)  # coseno de freq=3
        >>> X = dft(señal)
        >>> np.abs(X[3])   # Pico en la frecuencia 3
        32.0
    """
    N = len(x)
    # Solo calculamos N//2 + 1 coeficientes (simetría hermitiana para señales reales)
    X = np.zeros(N // 2 + 1, complex)

    for k in range(N // 2 + 1):
        for n in range(N):
            # Cada muestra x[n] contribuye con una fase e^(-2πi·k·n/N)
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X


def idft(X):
    """
    Calcula la Transformada Discreta de Fourier Inversa (IDFT) a partir de los
    coeficientes de Fourier de una señal real.

    Reconstruye la señal original x[n] en el dominio del tiempo a partir de sus
    coeficientes espectrales X[k]. Asume que la señal original era real, por lo
    que los coeficientes de entrada son solo la mitad positiva (N/2 + 1 valores).

    Fórmula:
        x[n] = (1/N) · [Σₖ₌₀^{N/2} X[k]·e^(2πi·k·n/N) + Σₖ₌₁^{N/2-1} X[k]*·e^(-2πi·k·n/N)]

    donde X[k]* es el conjugado complejo de X[k]. El segundo sumatorio reconstruye
    las frecuencias negativas a partir de la simetría hermitiana.

    Parámetros:
        X : array complejo - Los N/2 + 1 coeficientes de Fourier (salida de dft()).

    Retorna:
        array complejo - La señal reconstruida (la parte imaginaria debería ser ≈ 0
                         si la señal original era real).

    Ejemplo:
        >>> X = dft(señal)
        >>> señal_reconstruida = idft(X)
        >>> np.allclose(señal_reconstruida.real, señal)
        True
    """
    # Deducir el largo original N a partir del número de coeficientes
    N = (len(X) - 1) * 2
    Y = np.zeros(N, complex)

    for n in range(N):
        # Sumar las contribuciones de las frecuencias positivas (k = 0 a N/2)
        for k in range(N // 2 + 1):
            Y[n] += X[k] * np.exp(2j * np.pi * k * n / N)

        # Sumar las frecuencias negativas usando la simetría hermitiana:
        # X[-k] = X[k]* para señales reales, lo que equivale a e^(-2πi·k·n/N)
        for k in range(1, N // 2):
            Y[n] += X[k].conjugate() * np.exp(-2j * np.pi * k * n / N)

        # Normalizar dividiendo entre N
        Y[n] /= N

    return Y


def fft(x):
    """
    Calcula la Transformada Rápida de Fourier (FFT) usando el algoritmo de Cooley-Tukey.

    Es equivalente a la DFT pero mucho más eficiente: O(N·log N) en vez de O(N²).
    Funciona dividiendo recursivamente la señal en muestras pares e impares,
    calculando la FFT de cada mitad, y combinando los resultados.

    Requisito: el largo N de la señal debe ser una potencia de 2 (16, 32, 64, 128, ...).

    A diferencia de dft(), esta función retorna los N coeficientes completos
    (frecuencias positivas y negativas), igual que np.fft.fft().

    Parámetros:
        x : array o lista - La señal de entrada (N muestras, N debe ser potencia de 2).

    Retorna:
        lista de complejos - Los N coeficientes de Fourier.

    Ejemplo:
        >>> señal = np.cos(2 * np.pi * 3 * np.arange(64) / 64)
        >>> X = fft(señal)
        >>> len(X)
        64
    """
    N = len(x)
    if N <= 1:
        # Caso base: la FFT de un solo elemento es el elemento mismo
        return x
    else:
        # Dividir: FFT de las muestras en posiciones pares e impares
        even = fft(x[0::2])   # x[0], x[2], x[4], ...
        odd = fft(x[1::2])    # x[1], x[3], x[5], ...

        # Factores «twiddle»: W_N^k = e^(-2πi·k/N), que rotan las frecuencias impares
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

        # Combinar usando la «mariposa» de Cooley-Tukey:
        # X[k]       = E[k] + W·O[k]     (primera mitad)
        # X[k + N/2] = E[k] - W·O[k]     (segunda mitad)
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]


def ifft(X):
    """
    Calcula la Transformada Rápida de Fourier Inversa (IFFT).

    Reconstruye la señal original a partir de sus N coeficientes de Fourier completos
    (la salida de fft()). Se basa en la identidad de que la IFFT puede calcularse
    conjugando la entrada, aplicando la FFT, conjugando la salida y dividiendo entre N.

    Fórmula: x[n] = (1/N) · FFT(X*)* 

    Parámetros:
        X : lista de complejos - Los N coeficientes de Fourier (salida de fft()).

    Retorna:
        lista de complejos - La señal reconstruida en el dominio del tiempo.

    Ejemplo:
        >>> X = fft(señal)
        >>> señal_reconstruida = ifft(X)
        >>> np.allclose(señal_reconstruida, señal)
        True
    """
    N = len(X)
    # Paso 1: conjugar todos los coeficientes
    conjugada = [val.conjugate() for val in X]
    # Paso 2: aplicar la FFT directa a la señal conjugada
    resultado = fft(conjugada)
    # Paso 3: conjugar de nuevo y dividir entre N para normalizar
    return [val.conjugate() / N for val in resultado]


# ============================================================================
# PLANTILLA: CARGA DE DATOS DESDE ARCHIVOS
# ============================================================================
#
# --- Con NumPy ---
#
#   # .txt o .dat (columnas separadas por espacios o tabuladores)
#   datos = np.loadtxt("archivo.txt")
#   datos = np.loadtxt("archivo.dat")
#
#   # .csv (separado por comas)
#   datos = np.loadtxt("archivo.csv", delimiter=",")
#
#   # Si tiene encabezado (saltarse la primera fila)
#   datos = np.loadtxt("archivo.txt", skiprows=1)
#
#   # Extraer columnas individuales
#   x = datos[:, 0]       # primera columna
#   y = datos[:, 1]       # segunda columna
#   sigma = datos[:, 2]   # tercera columna (incertidumbres)
#
#   # Desempaquetar varias columnas a la vez
#   x, y, sigma = np.loadtxt("archivo.csv", delimiter=",", skiprows=1, unpack=True)
#
#
# --- Con Pandas ---
#
#   import pandas as pd
#
#   # .csv
#   df = pd.read_csv("archivo.csv")
#
#   # .txt o .dat (separado por espacios/tabs)
#   df = pd.read_csv("archivo.txt", sep=r"\s+")
#   df = pd.read_csv("archivo.dat", sep=r"\s+")
#
#   # Si no tiene encabezado
#   df = pd.read_csv("archivo.csv", header=None, names=["x", "y", "sigma"])
#
#   # Extraer columnas como arrays de NumPy
#   x = df["x"].values
#   y = df["y"].values
#   sigma = df["sigma"].values
#
#   # O por posición
#   x = df.iloc[:, 0].values
#   y = df.iloc[:, 1].values
#
#
# --- Ejemplo completo ---
#
#   import numpy as np
#   import fcl1109 as fc
#
#   # Cargar datos
#   x, y, sigma = np.loadtxt("datos.csv", delimiter=",", skiprows=1, unpack=True)
#
#   # Ajuste lineal y = a0 + a1*x
#   S   = np.sum(1 / sigma**2)
#   Sx  = np.sum(x / sigma**2)
#   Sxx = np.sum(x**2 / sigma**2)
#   Sy  = np.sum(y / sigma**2)
#   Sxy = np.sum(x * y / sigma**2)
#
#   A = np.array([[S, Sx], [Sx, Sxx]])
#   b = np.array([Sy, Sxy])
#   a0, a1 = mt.minimos_cuadrados(A, b)
#
#   y_ajuste = a0 + a1 * x
#   chi2 = mt.chi_cuadrada(y, y_ajuste, sigma)
#   print(f"y = {a0:.4f} + {a1:.4f}·x,  χ² = {chi2:.4f}")


# ============================================================================
# PLANTILLA: GRÁFICAS CON MATPLOTLIB
# ============================================================================
#
# import matplotlib.pyplot as plt
#
#
# --- Gráfica básica (línea) ---
#
#   plt.figure(figsize=(8, 5))
#   plt.plot(x, y, "b-", linewidth=1.5, label="Datos")
#   plt.xlabel("Tiempo $t$ [s]", fontsize=13)
#   plt.ylabel("Posición $x$ [m]", fontsize=13)
#   plt.title("Posición vs Tiempo", fontsize=14)
#   plt.legend(fontsize=12)
#   plt.grid(True, alpha=0.3)
#   plt.tight_layout()
#   plt.savefig("grafica.png", dpi=150)
#   plt.show()
#
#
# --- Datos experimentales con barras de error + ajuste ---
#
#   plt.figure(figsize=(8, 5))
#   plt.errorbar(x, y, yerr=sigma, fmt="ko", markersize=4,
#                capsize=3, label="Datos experimentales")
#   plt.plot(x, y_ajuste, "r-", linewidth=1.5,
#            label=f"Ajuste: $y = {a0:.3f} + {a1:.3f}x$")
#   plt.xlabel("$x$ [unidad]", fontsize=13)
#   plt.ylabel("$y$ [unidad]", fontsize=13)
#   plt.title("Ajuste por mínimos cuadrados", fontsize=14)
#   plt.legend(fontsize=12)
#   plt.grid(True, alpha=0.3)
#   plt.tight_layout()
#   plt.savefig("ajuste.png", dpi=150)
#   plt.show()
#
#
# --- Varias curvas en una gráfica ---
#
#   plt.figure(figsize=(8, 5))
#   plt.plot(t, x1, "b-", linewidth=1.5, label="$x_1(t)$")
#   plt.plot(t, x2, "r--", linewidth=1.5, label="$x_2(t)$")
#   plt.plot(t, x3, "g:", linewidth=1.5, label="$x_3(t)$")
#   plt.xlabel("Tiempo $t$ [s]", fontsize=13)
#   plt.ylabel("Amplitud [m]", fontsize=13)
#   plt.title("Comparación de soluciones", fontsize=14)
#   plt.legend(fontsize=12)
#   plt.grid(True, alpha=0.3)
#   plt.tight_layout()
#   plt.savefig("comparacion.png", dpi=150)
#   plt.show()
#
#
# --- Subplots (varias gráficas en una figura) ---
#
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
#   ax1.plot(t, x, "b-", linewidth=1.5)
#   ax1.set_xlabel("$t$ [s]", fontsize=13)
#   ax1.set_ylabel("$x$ [m]", fontsize=13)
#   ax1.set_title("Posición", fontsize=14)
#   ax1.grid(True, alpha=0.3)
#
#   ax2.plot(t, v, "r-", linewidth=1.5)
#   ax2.set_xlabel("$t$ [s]", fontsize=13)
#   ax2.set_ylabel("$v$ [m/s]", fontsize=13)
#   ax2.set_title("Velocidad", fontsize=14)
#   ax2.grid(True, alpha=0.3)
#
#   fig.suptitle("Oscilador armónico", fontsize=15)
#   fig.tight_layout()
#   fig.savefig("subplots.png", dpi=150)
#   plt.show()
#
#
# --- Espectro de frecuencias (DFT/FFT) ---
#
#   X = mt.dft(señal)               # o mt.fft(señal)
#   freqs = np.arange(len(X)) / N   # frecuencias normalizadas (ciclos/muestra)
#   # freqs = np.arange(len(X)) * fs / N   # si se conoce la tasa de muestreo fs [Hz]
#
#   plt.figure(figsize=(8, 5))
#   plt.stem(freqs, np.abs(X), linefmt="b-", markerfmt="bo", basefmt="k-")
#   plt.xlabel("Frecuencia [Hz]", fontsize=13)
#   plt.ylabel("$|X(f)|$", fontsize=13)
#   plt.title("Espectro de frecuencias", fontsize=14)
#   plt.grid(True, alpha=0.3)
#   plt.tight_layout()
#   plt.savefig("espectro.png", dpi=150)
#   plt.show()
#
#
# --- Diagrama de fase (EDOs vectoriales) ---
#
#   plt.figure(figsize=(6, 6))
#   plt.plot(x_lista, v_lista, "b-", linewidth=1)
#   plt.plot(x_lista[0], v_lista[0], "go", markersize=8, label="Inicio")
#   plt.xlabel("Posición $x$ [m]", fontsize=13)
#   plt.ylabel("Velocidad $v$ [m/s]", fontsize=13)
#   plt.title("Espacio de fases", fontsize=14)
#   plt.legend(fontsize=12)
#   plt.grid(True, alpha=0.3)
#   plt.axis("equal")
#   plt.tight_layout()
#   plt.savefig("fase.png", dpi=150)
#   plt.show()
#
#
# --- Referencia rápida de formatos ---
#
#   Colores:  "b" azul, "r" rojo, "g" verde, "k" negro, "m" magenta, "c" cyan
#   Líneas:   "-" sólida, "--" discontinua, ":" punteada, "-." punto-raya
#   Marcas:   "o" círculo, "s" cuadrado, "^" triángulo, "x" cruz, "+" más
#   Combinar: "bo-" = azul + círculos + línea,  "r--" = rojo + discontinua
#
#   Texto con LaTeX: usar $ $ para ecuaciones → "$F = ma$", "$\\alpha$", "$x^2$"


# ============================================================================
# REFERENCIA: LISTAS DE PYTHON vs ARRAYS DE NUMPY
# ============================================================================
#
# En física computacional se trabaja casi siempre con arrays de NumPy, no con
# listas de Python. La diferencia no es solo de rendimiento: las operaciones
# aritméticas se comportan de manera distinta en cada caso.
#
#
# --- Diferencia fundamental en aritmética ---
#
#   lista = [1, 2, 3]
#   array = np.array([1, 2, 3])
#
#   lista * 2          →  [1, 2, 3, 1, 2, 3]   # duplica la lista (concatenación)
#   array * 2          →  array([2, 4, 6])       # multiplica cada elemento
#
#   lista + lista      →  [1, 2, 3, 1, 2, 3]   # concatena
#   array + array      →  array([2, 4, 6])       # suma elemento a elemento
#
#   lista ** 2         →  TypeError              # no soportado
#   array ** 2         →  array([1, 4, 9])       # potencia elemento a elemento
#
#   np.sin(lista)      →  funciona (NumPy convierte la lista internamente)
#   np.sin(array)      →  funciona, y es más eficiente
#
# Regla: si vas a hacer aritmética con los datos, usa arrays desde el principio.
#
#
# --- Construcción de arrays ---
#
#   # Desde una lista (conversión explícita)
#   x = np.array([1.0, 2.0, 3.0])
#
#   # N puntos equiespaciados en [a, b] - el más usado para grillas de tiempo/espacio
#   t = np.linspace(0.0, 10.0, 1000)    # incluye ambos extremos
#
#   # Array de ceros para pre-asignar (ver patrón de integración abajo)
#   x = np.zeros(N)
#   x = np.zeros(N, complex)            # si los valores son complejos
#
#   # Array de índices enteros
#   k = np.arange(N)                    # [0, 1, 2, ..., N-1]
#   k = np.arange(1, N+1)              # [1, 2, ..., N]
#
#
# --- Patrón correcto para integrar EDOs (pre-asignación) ---
#
# Al integrar una EDO con Euler o RK4, el tamaño del resultado se conoce antes
# del bucle. Pre-asignar el array con np.zeros es más eficiente y claro que
# ir acumulando resultados con list.append().
#
#   N = int((tf - t0) / dt)
#   t_vals = np.zeros(N + 1)
#   x_vals = np.zeros(N + 1)
#   v_vals = np.zeros(N + 1)
#
#   estado = np.array([x0, v0])    # vector de estado inicial
#   t_vals[0] = t0
#   x_vals[0] = x0
#   v_vals[0] = v0
#
#   for i in range(N):
#       estado = mt.rk4(t_vals[i], dt, estado, f)
#       t_vals[i+1] = t_vals[i] + dt
#       x_vals[i+1] = estado[0]
#       v_vals[i+1] = estado[1]
#
# Evitar el patrón con append:
#
#   t_vals = []
#   x_vals = []
#   for i in range(N):             # ← más lento: reasigna memoria en cada paso
#       x_vals.append(...)
#
#
# --- Vectores de estado para sistemas de EDOs ---
#
# Cuando la EDO describe un sistema con varias variables (posición + velocidad,
# o múltiples osciladores), el estado se representa como un array y la función
# f(t, s) devuelve un array de la misma forma:
#
#   def f(t, s):
#       x, v = s[0], s[1]                         # desempaquetar
#       return np.array([v, -omega**2 * x])        # [dx/dt, dv/dt]
#
#   estado = np.array([x0, v0])
#   estado = mt.rk4(t, dt, estado, f)             # rk4 acepta arrays directamente
#   x_vals[i+1] = estado[0]
#   v_vals[i+1] = estado[1]
#
#
# --- Operaciones vectorizadas post-integración ---
#
# Una vez que los arrays están llenos, las cantidades derivadas (energía, módulo,
# fase, etc.) se calculan sobre el array completo, sin loops:
#
#   E_cin = 0.5 * m * v_vals**2              # energía cinética en cada instante
#   E_pot = 0.5 * k * x_vals**2             # energía potencial
#   E_tot = E_cin + E_pot                    # energía total (array completo)
#
#   amplitud = np.sqrt(x_vals**2 + y_vals**2)
#   fase     = np.arctan2(y_vals, x_vals)
#
# Esto es más rápido y legible que calcular la energía dentro del bucle de integración.
#
#
# --- Indexación y slicing ---
#
#   a = np.array([10, 20, 30, 40, 50])
#
#   a[0]        →  10            # primer elemento
#   a[-1]       →  50            # último elemento
#   a[1:3]      →  [20, 30]     # desde índice 1 hasta 2 (el 3 no se incluye)
#   a[::2]      →  [10, 30, 50] # uno de cada dos (stride)
#   a[1::2]     →  [20, 40]     # impares (stride desde índice 1)
#
# Para arrays 2D (por ejemplo, datos cargados de un archivo):
#
#   D = np.loadtxt("datos.dat")
#   t     = D[:, 0]    # toda la primera columna
#   y     = D[:, 1]    # toda la segunda columna
#   sigma = D[:, 2]    # toda la tercera columna
#
#   D[0, :]    # primera fila completa
#   D[:, -1]   # última columna completa
#
#
# --- Funciones de agregación ---
#
#   np.sum(a)       # suma de todos los elementos
#   np.mean(a)      # promedio
#   np.max(a)       # valor máximo
#   np.min(a)       # valor mínimo
#   np.abs(a)       # valor absoluto elemento a elemento
#   np.sqrt(a)      # raíz cuadrada elemento a elemento
#
#   np.sum(a**2)            # suma de cuadrados
#   np.sum(a / sigma**2)    # suma ponderada (aparece en mínimos cuadrados)
#
#
# --- Error frecuente: pasar una lista donde se espera un array ---
#
# Las funciones de fcl1109.py (trapecio, simpson, etc.) evalúan f(x) donde x
# puede ser un array de NumPy. Si f está definida con operaciones de lista,
# puede fallar o dar resultados incorrectos:
#
#   # Incorrecto - no acepta arrays:
#   def f(x):
#       return x**2 + 1              # esto sí funciona con arrays (NumPy lo maneja)
#
#   def f(x):
#       return [xi**2 + 1 for xi in x]   # devuelve lista, no array - puede romper
#                                          # operaciones posteriores
#
#   # Correcto - retorna array directamente:
#   f = lambda x: x**2 + 1               # NumPy extiende ** y + a arrays
#   f = lambda x: np.exp(-x**2)          # np.exp acepta arrays
#
# Si la función tiene ramas condicionales, usar np.where en lugar de if/else:
#
#   # Incorrecto para arrays:
#   def f(x):
#       if x > 0:
#           return x**2
#       else:
#           return 0.0
#
#   # Correcto:
#   def f(x):
#       return np.where(x > 0, x**2, 0.0)