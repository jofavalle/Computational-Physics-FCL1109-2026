# Practica Numerica 1 - Problema 2
# Fisica Computacional, Ciclo I 2026
# Carnet: AV18012
# Energia de un gas de fotones con densidad de estados g(w) = A w^2 exp(-w/wc)
# U(T) = integral de 0 a inf de  hbar*w/(exp(beta*hbar*w)-1) * g(w) dw

import numpy as np
from scipy import integrate
from scipy.constants import hbar, k as kB
import matplotlib.pyplot as plt

A = 1.0
wc = 5e13
temperaturas = [3.0, 300.0, 6000.0]


# integrando. en w=0 queda 0/0 pero el limite vale 0, asi que lo pongo en 0
# uso expm1 para que exp(x)-1 no pierda precision cuando x es chico
def integrando(w, T):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    beta = 1.0 / (kB*T)
    f = np.zeros_like(w)
    m = w > 0
    ww = w[m]
    f[m] = hbar*ww * (A*ww**2*np.exp(-ww/wc)) / np.expm1(beta*hbar*ww)
    return f if f.size > 1 else f[0]


# limite superior: el integrando decae como exp(-w/wc)*exp(-w/wth),
# con wth = kB T/hbar. Tomo 50 veces la escala combinada, ahi ya es ~0
def w_max(T):
    wth = kB*T/hbar
    return 50 * (wc*wth/(wc+wth))


def trapecio(f, a, b, N):
    x = np.linspace(a, b, N+1)
    y = f(x)
    h = (b-a)/N
    return h*(0.5*y[0] + y[1:-1].sum() + 0.5*y[-1])


def simpson(f, a, b, N):
    if N % 2 == 1:   # Simpson necesita N par
        N += 1
    x = np.linspace(a, b, N+1)
    y = f(x)
    h = (b-a)/N
    return h/3*(y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())


def monte_carlo(f, a, b, N, rng):
    x = rng.uniform(a, b, N)
    return (b-a)*f(x).mean()


rng = np.random.default_rng(12345)
N = 2000
Nmc = 2000000
U = {}

for T in temperaturas:
    b = w_max(T)
    f = lambda w: integrando(w, T)
    ref = integrate.quad(f, 0, b, limit=200)[0]   # lo uso como valor exacto
    It = trapecio(f, 0, b, N)
    Is = simpson(f, 0, b, N)
    Im = monte_carlo(f, 0, b, Nmc, rng)
    U[T] = ref
    print("T =", T, "K")
    print("  quad     =", ref)
    print("  trapecio =", It, " err rel =", abs(It-ref)/abs(ref))
    print("  simpson  =", Is, " err rel =", abs(Is-ref)/abs(ref))
    print("  monte c. =", Im, " err rel =", abs(Im-ref)/abs(ref))

# grafico el integrando para cada T y la energia total vs T
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for T, c in zip(temperaturas, ['tab:blue', 'tab:orange', 'tab:red']):
    w = np.linspace(1, w_max(T), 2000)
    a1.plot(w, integrando(w, T), color=c, label="T = %g K" % T)
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xlabel('w [1/s]'); a1.set_ylabel('integrando')
a1.set_title('Integrando para cada temperatura')
a1.grid(alpha=0.3, which='both'); a1.legend()

Ts = np.array(temperaturas)
Us = np.array([U[T] for T in temperaturas])
a2.loglog(Ts, Us, 'o-', color='tab:green')
a2.set_xlabel('T [K]'); a2.set_ylabel('U(T)')
a2.set_title('Energia total vs temperatura')
a2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig("P2_energia.png", dpi=130)
plt.close(fig)
