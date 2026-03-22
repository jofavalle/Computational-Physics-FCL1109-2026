"""
Librería Física Común Centroamericana (LFCC) — «manteca.py»
============================================================

Librería de métodos numéricos desarrollada para el curso FCL1109.
Contiene herramientas para:

  1. Derivación numérica (adelante, central, n-ésima derivada)
  2. Integración numérica (trapecio, Simpson, Monte Carlo, cambio de variable)
  3. Resolución de EDOs (Euler, Runge-Kutta 4)
  4. Ajuste de datos (mínimos cuadrados, chi cuadrada)
  5. Búsqueda de raíces (Newton-Raphson)
  6. Análisis de Fourier (DFT, IDFT, FFT, IFFT)

Uso básico:
    import manteca as mt
    resultado = mt.derivada_central(np.sin, 1.0, 1e-5)
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
        f : función    — La función a derivar. Debe aceptar un número y devolver un número.
        x : float      — El punto donde queremos evaluar la derivada.
        h : float      — El tamaño del paso. Valores típicos: 1e-5 a 1e-8.

    Retorna:
        float — Aproximación numérica de f'(x).

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
        f : función    — La función a derivar.
        x : float      — El punto donde queremos evaluar la derivada.
        h : float      — El tamaño del paso. Valores típicos: 1e-5 a 1e-8.

    Retorna:
        float — Aproximación numérica de f'(x).

    Ejemplo:
        >>> derivada_central(lambda x: x**3, 2.0, 1e-5)
        12.000000000000  # ≈ 3·(2)² = 12
    """
    return (f(x + h/2) - f(x - h/2)) / h


def derivada_nesima(f, x, n):
    """
    Calcula la n-ésima derivada de f en el punto x de forma recursiva.

    Utiliza la derivada central internamente. Para calcular la segunda derivada,
    primero construye una nueva función que representa la primera derivada,
    y luego le aplica la derivada central otra vez. Y así sucesivamente.

    Usa el paso global _h_default (= 1e-5) definido al inicio del archivo.

    Parámetros:
        f : función    — La función a derivar.
        x : float      — El punto donde evaluar la derivada.
        n : int        — El orden de la derivada (0 = la función misma, 1 = primera, 2 = segunda, etc.)

    Retorna:
        float — Aproximación numérica de f⁽ⁿ⁾(x).

    Nota:
        La precisión disminuye conforme n aumenta, porque los errores numéricos
        se acumulan en cada nivel de recursión.

    Ejemplo:
        >>> derivada_nesima(lambda x: x**3, 2.0, 1)   # Primera derivada: 3x² = 12
        12.0
        >>> derivada_nesima(lambda x: x**3, 2.0, 2)   # Segunda derivada: 6x = 12
        12.0
    """
    if n == 0:
        # Caso base: la «derivada de orden 0» es simplemente evaluar la función
        return f(x)
    else:
        # Caso recursivo: construimos g(y) = f'(y) usando derivada central,
        # y luego calculamos la (n-1)-ésima derivada de g
        return derivada_nesima(lambda y: derivada_central(f, y, _h_default), x, n - 1)


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
        f : función    — La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      — Límite inferior de integración.
        b : float      — Límite superior de integración.
        n : int        — Número de subintervalos (más grande = más preciso).

    Retorna:
        float — Aproximación numérica de ∫ₐᵇ f(x)dx.

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
        f : función    — La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      — Límite inferior de integración.
        b : float      — Límite superior de integración.
        n : int        — Número de subintervalos (se ajusta a par si es impar).

    Retorna:
        float — Aproximación numérica de ∫ₐᵇ f(x)dx.

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
        f : función    — La función a integrar. Debe poder evaluarse con arrays de NumPy.
        a : float      — Límite inferior de integración.
        b : float      — Límite superior de integración.
        N : int        — Número de puntos aleatorios (más grande = más preciso pero más lento).

    Retorna:
        float — Aproximación numérica de ∫ₐᵇ f(x)dx.

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
        z : float — Valor en el intervalo (0, 1).

    Retorna:
        float — El valor x = z/(1-z) en el intervalo (0, ∞).

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
        z : float — Valor en el intervalo (0, 1).

    Retorna:
        float — El jacobiano 1/(1-z)².

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
        f : función    — La función original que queremos integrar de 0 a ∞.
        z : float      — Punto en (0, 1) donde evaluar el integrando transformado.

    Retorna:
        float — Valor del integrando transformado en z.

    Ejemplo:
        Para calcular ∫₀^∞ e^(-x) dx = 1:
        >>> eps = 1e-6
        >>> simpson(lambda z: integrando_transformado(np.exp(-x), z), eps, 1-eps, 1000)
        0.999999   # ≈ 1
    """
    x = cambio_variable(z)
    return f(x) * jacobiano(z)


# ============================================================================
# 3. RESOLUCIÓN DE ECUACIONES DIFERENCIALES ORDINARIAS (EDOs)
# ============================================================================

def euler(f, x, h):
    """
    Realiza UN paso del método de Euler para resolver una EDO.

    El método de Euler es el integrador numérico más simple. Dada una EDO
    de la forma dx/dt = f(x), avanza la solución un paso temporal h usando:

        x(t + h) ≈ x(t) + h · f(x(t))

    Es un método de orden 1: el error por paso es O(h²) y el error acumulado es O(h).
    Para mayor precisión, usar rk4().

    Nota: esta función da UN solo paso. Para integrar en un intervalo completo,
    hay que llamarla repetidamente en un bucle.

    Parámetros:
        f : función    — El lado derecho de la EDO dx/dt = f(x). Recibe x, devuelve dx/dt.
        x : float o array — El estado actual del sistema.
        h : float      — El tamaño del paso temporal.

    Retorna:
        float o array — El nuevo estado x(t + h).

    Ejemplo:
        Para dx/dt = -x con x(0) = 1, integrar hasta t = 1 con dt = 0.001:
        >>> x = 1.0
        >>> for _ in range(1000):
        ...     x = euler(lambda x: -x, x, 0.001)
        >>> print(x)
        0.3677   # ≈ e^(-1) = 0.3679
    """
    return x + h * f(x)


def rk4(t, h, x, f):
    """
    Realiza UN paso del método de Runge-Kutta de orden 4 (RK4) para resolver una EDO.

    RK4 es el «caballo de batalla» de los integradores numéricos. Evalúa la función
    f en 4 puntos estratégicos dentro del paso y combina esos valores para lograr
    un error por paso de orden O(h⁵), mucho mayor que Euler.

    Dada una EDO de la forma dx/dt = f(t, x), el algoritmo calcula:
        k₁ = h · f(t, x)
        k₂ = h · f(t + h/2, x + k₁/2)
        k₃ = h · f(t + h/2, x + k₂/2)
        k₄ = h · f(t + h, x + k₃)
        x(t + h) = x(t) + (k₁ + 2k₂ + 2k₃ + k₄) / 6

    Funciona tanto para EDOs escalares (x es un float) como para sistemas de EDOs
    (x es un array de NumPy), lo que permite resolver sistemas acoplados.

    Parámetros:
        t : float          — El tiempo actual.
        h : float          — El tamaño del paso temporal.
        x : float o array  — El estado actual (escalar para 1 EDO, array para sistemas).
        f : función        — El lado derecho de la EDO. Debe tener firma f(t, x) → dx/dt.

    Retorna:
        float o array — El nuevo estado x(t + h).

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
        A : array 2D (n×n)  — La matriz de coeficientes del sistema normal.
        b : array 1D (n)    — El vector del lado derecho.

    Retorna:
        array 1D (n) — Los coeficientes del ajuste [a₀, a₁, ..., aₙ₋₁].

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
        y_observado : array — Valores medidos experimentalmente.
        y_ajustado  : array — Valores predichos por el modelo.
        sigma       : array — Incertidumbres de cada medición.

    Retorna:
        float — El valor de χ².

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
# 5. BÚSQUEDA DE RAÍCES – NEWTON-RAPHSON
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
        f    : función — La función cuya raíz buscamos.
        x    : float   — Estimación inicial de la raíz.
        dx   : float   — Paso para calcular la derivada numérica (típico: 1e-6).
        eps  : float   — Tolerancia: el algoritmo se detiene cuando |f(x)| ≤ eps.
        Nmax : int     — Número máximo de iteraciones (protección contra no-convergencia).

    Retorna:
        float — La raíz aproximada.

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
        x : array — La señal de entrada (N muestras, valores reales).

    Retorna:
        array complejo — Los N/2 + 1 coeficientes de Fourier X[k].

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
        X : array complejo — Los N/2 + 1 coeficientes de Fourier (salida de dft()).

    Retorna:
        array complejo — La señal reconstruida (la parte imaginaria debería ser ≈ 0
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
        x : array o lista — La señal de entrada (N muestras, N debe ser potencia de 2).

    Retorna:
        lista de complejos — Los N coeficientes de Fourier.

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
    (la salida de fft()). Usa un truco elegante: la IFFT se puede calcular
    conjugando la entrada, aplicando la FFT, conjugando la salida y dividiendo entre N.

    Fórmula: x[n] = (1/N) · FFT(X*)* 

    Parámetros:
        X : lista de complejos — Los N coeficientes de Fourier (salida de fft()).

    Retorna:
        lista de complejos — La señal reconstruida en el dominio del tiempo.

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