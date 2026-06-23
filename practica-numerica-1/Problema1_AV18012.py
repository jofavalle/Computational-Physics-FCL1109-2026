# Practica Numerica 1 - Problema 1
# Fisica Computacional, Ciclo I 2026
# Carnet: AV18012
# Polinomios de Legendre con la formula de Rodrigues usando diferencias centradas

import numpy as np
import matplotlib.pyplot as plt
from math import factorial


# (a) coeficiente principal a_n = (2n)! / (2^n (n!)^2)
def coef_principal(n):
    return factorial(2*n) / (2.0**n * factorial(n)**2)


# polinomios de Legendre analiticos (forma cerrada, de tabla) para comparar
P_analitico = {
    1: lambda x: x,
    2: lambda x: (3*x**2 - 1) / 2,
    3: lambda x: (5*x**3 - 3*x) / 2,
    4: lambda x: (35*x**4 - 30*x**2 + 3) / 8,
    5: lambda x: (63*x**5 - 70*x**3 + 15*x) / 8,
    6: lambda x: (231*x**6 - 315*x**4 + 105*x**2 - 5) / 16,
    7: lambda x: (429*x**7 - 693*x**5 + 315*x**3 - 35*x) / 16,
    8: lambda x: (6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35) / 128,
}


# diferencia centrada de primer orden: f'(x) = (f(x+h)-f(x-h))/(2h)
def dif_central(f, h):
    return lambda x: (f(x+h) - f(x-h)) / (2*h)


# la derivada de orden n es aplicar n veces la diferencia central
# (cada derivada usa la anterior)
def derivada_n(f, n, h):
    g = f
    for _ in range(n):
        g = dif_central(g, h)
    return g


# P_n por Rodrigues: derivo n veces (x^2-1)^n y divido entre 2^n n!
def P_rodrigues(n, h):
    f = lambda x: (x**2 - 1.0)**n
    dn = derivada_n(f, n, h)
    return lambda x: dn(x) / (2.0**n * factorial(n))


def comparar(h, archivo):
    x = np.linspace(-1, 1, 401)
    fig, ax = plt.subplots(4, 2, figsize=(11, 14))
    ax = ax.ravel()
    print("h =", h)
    for n in range(1, 9):
        Pn = P_rodrigues(n, h)(x)
        ana = P_analitico[n](x)
        err = np.max(np.abs(Pn - ana))
        print("P%d  error max = %.3e" % (n, err))
        ax[n-1].plot(x, ana, 'k-', lw=2, label='analitico')
        ax[n-1].plot(x, Pn, 'r--', lw=1.2, label='Rodrigues')
        ax[n-1].set_title("P%d(x), error max = %.1e" % (n, err))
        ax[n-1].set_xlabel('x')
        ax[n-1].grid(alpha=0.3)
        ax[n-1].legend(fontsize=8)
    fig.suptitle("Polinomios de Legendre por Rodrigues (h = %g)" % h)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(archivo, dpi=130)
    plt.close(fig)


# coeficientes principales (parte a)
print("coeficientes principales a_n:")
for n in range(1, 9):
    print("a_%d = %.6f" % (n, coef_principal(n)))

# (b) con h = 0.01 todas las P_n salen bien
comparar(0.01, "P1_legendre_h0.01.png")

# (c) con h = 1e-3 las de orden bajo mejoran pero las de orden alto se
# desestabilizan: al dividir entre (2h)^n el redondeo se amplifica
comparar(1e-3, "P1_legendre_h0.001.png")
