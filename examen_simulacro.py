"""
╔══════════════════════════════════════════════════════════════════════╗
║              EXAMEN SIMULACRO — FÍSICA COMPUTACIONAL               ║
║                         FCL1109 · 2026                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Duración : 1 hora 30 minutos                                     ║
║  Permitido: fcl1109.py (como fc), numpy (uso básico), matplotlib   ║
║  Archivos : datos_decaimiento.csv, senal_ruido.csv                 ║
║                                                                    ║
║  Instrucciones generales:                                          ║
║  - Cada problema indica lo que se debe calcular, graficar o        ║
║    imprimir.                                                       ║
║  - Use print() para reportar resultados numéricos.                 ║
║  - Guarde todas las gráficas con plt.savefig().                    ║
║  - Comente brevemente cada paso de su código.                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import fcl1109 as fc

# ====================================================================
# PROBLEMA 1 — Potencial de Lennard-Jones          (25 pts · ~20 min)
# ====================================================================
#
# El potencial de Lennard-Jones modela la interacción entre dos átomos
# neutros:
#
#          V(r) = 4ε [ (σ/r)^12  −  (σ/r)^6 ]
#
# con  ε = 0.0103 eV  y  σ = 3.40 Å  (parámetros del argón).
#
# a) Defina la función V(r).
#
# b) Use Newton-Raphson para encontrar la distancia de equilibrio r_eq,
#    es decir, el punto donde la fuerza F(r) = −dV/dr = 0.
#    (Sugerencia: busque la raíz de F(r) con x₀ = 3.5 Å.)
#    Imprima r_eq con al menos 6 decimales.
#
# c) Calcule la segunda derivada V''(r_eq) usando derivada_enesima.
#    Esto equivale a la "constante de resorte" efectiva κ del enlace.
#    Imprima κ en eV/Å².
#
# d) Grafique V(r) en el rango r ∈ [3.0, 7.0] Å.
#    Marque con un punto rojo la posición de equilibrio (r_eq, V(r_eq)).
#    Etiquete los ejes y guarde la gráfica como "p1_lennard_jones.png".
#


def V(r):
    """Potencial de Lennard-Jones para el argón."""
    epsilon = 0.0103  # eV
    sigma = 3.40      # Å
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

derivada_V = lambda r: fc.derivada_enesima(V, r, 1)

derivada_2_V = lambda r: fc.derivada_enesima(V, r, 2)

r_eq = fc.newton_raphson(derivada_V, 3.5, 1e-6, 1e-6, 1000)

print(f"Distancia de equilibrio r_eq = {r_eq:.6f} Å")

kappa = derivada_2_V(r_eq)
print(f"Constante de resorte efectiva κ = {kappa:.6f} eV/Å²")

r_values = np.linspace(3.0, 7.0, 400)
V_values = V(r_values)

plt.figure(figsize=(8, 5))
plt.plot(r_values, V_values, label='V(r)')
plt.plot(r_eq, V(r_eq), 'ro', label='Equilibrio')
plt.xlabel('Distancia r (Å)')
plt.ylabel('Potencial V(r) (eV)')
plt.title('Potencial de Lennard-Jones para el Argón')
plt.legend()
plt.grid()
plt.show()

# ====================================================================
# PROBLEMA 2 — Distribución de Maxwell-Boltzmann   (25 pts · ~20 min)
# ====================================================================
#
# La distribución de rapideces de Maxwell-Boltzmann es:
#
#      f(v) = 4π · (m / 2πkT)^(3/2) · v² · exp(−mv² / 2kT)
#
# Use:  m = 6.63 × 10⁻²⁶ kg  (argón),  T = 300 K,
#       k_B = 1.381 × 10⁻²³ J/K.
#
# a) Defina f(v) y verifique que está normalizada, es decir, que
#    ∫₀^∞ f(v) dv = 1 usando integral_impropia.
#    Imprima el resultado.

m = 6.63e-26  # kg
T = 300       # K
k_B = 1.381e-23  # J/K

def f(v):
    """Distribución de Maxwell-Boltzmann para el argón a T=300K."""
    prefactor = 4 * np.pi * (m / (2 * np.pi * k_B * T))**(3/2)
    return prefactor * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

normalization = fc.integral_impropia(f, 0, np.inf, 10000)
print(f"Normalización de f(v) = {normalization:.6f}")

#
# b) Calcule la rapidez más probable v_p (donde f(v) es máxima)
#    usando Newton-Raphson sobre f'(v) = 0.
#    (Sugerencia: x₀ = 300 m/s.)
#    Imprima v_p y compárelo con el valor analítico v_p = √(2kT/m).

derivada_f = lambda v: fc.derivada_enesima(f, v, 1)
v_p = fc.newton_raphson(derivada_f, 300, 1e-6, 1e-6, 1000)
v_p_analitico = np.sqrt(2 * k_B * T / m)
print(f"Rapidez más probable v_p = {v_p:.6f} m/s")
print(f"Valor analítico v_p = {v_p_analitico:.6f} m/s")

#
# c) Calcule la probabilidad de que una partícula tenga rapidez
#    v > 600 m/s, es decir P = ∫₆₀₀^∞ f(v) dv.
#    Imprima P.

P = fc.integral_impropia(f, 600, np.inf, 10000)
print(f"Probabilidad de v > 600 m/s: P = {P:.6f} ")

#
# d) Grafique f(v) en el rango v ∈ [0, 1000] m/s.
#    Sombree (con plt.fill_between) la región v > 600 m/s.
#    Marque v_p con una línea vertical punteada.
#    Guarde como "p2_maxwell.png".
#

v_values = np.linspace(0, 1000, 400)
f_values = f(v_values)

plt.figure(figsize=(8, 5))
plt.plot(v_values, f_values, label='f(v)')
plt.fill_between(v_values, f_values, where=(v_values > 600), color='orange', alpha=0.5, label='v > 600 m/s')
plt.axvline(v_p, color='red', linestyle='--', label=f'v_p = {v_p:.1f} m/s')
plt.xlabel('Rapidez v (m/s)')
plt.ylabel('f(v)')
plt.title('Distribución de Maxwell-Boltzmann para el Argón a T=300K')
plt.legend()
plt.grid()
plt.show()

# --- Escriba su código aquí ---


# ====================================================================
# PROBLEMA 3 — Oscilador armónico amortiguado      (25 pts · ~25 min)
# ====================================================================
#
# Un oscilador amortiguado obedece la ecuación:
#
#      d²x/dt² + 2γ dx/dt + ω₀² x = 0
#
# con  ω₀ = 2π rad/s  (frecuencia natural)  y  γ = 0.3 s⁻¹ (amortiguamiento).
#
# Condiciones iniciales:  x(0) = 1.0 m ,  v(0) = dx/dt(0) = 0 m/s.
#
# a) Reescriba la EDO de segundo orden como un sistema de dos EDOs
#    de primer orden:
#        dx/dt = v
#        dv/dt = −2γv − ω₀²x
#    Defina la función f(t, estado) que recibe t y estado = [x, v]
#    y devuelve [dx/dt, dv/dt].

def fs(t, estado):
    """Derivadas para el oscilador amortiguado."""
    x, v = estado
    dxdt = v
    dvdt = -2 * 0.3 * v - (2 * np.pi)**2 * x
    return np.array([dxdt, dvdt])

#
# b) Integre desde t = 0 hasta t = 15 s usando rk4 con dt = 0.01.
#    Almacene los arreglos de t, x(t) y v(t).

N_pasos = int((15-0)/0.01) 

t_vals = np.zeros(N_pasos+1)
x_vals = np.zeros(N_pasos+1)
v_vals = np.zeros(N_pasos+1)

estado = np.array([1.0, 0.0])  # x(0) = 1.0 m, v(0) = 0 m/s

t_vals[0] = 0
x_vals[0] = estado[0]
v_vals[0] = estado[1]

for i in range(N_pasos):
    #estado = fc.rk4(fs, t_vals[i], estado, 0.01)
    estado = fc.euler(fs, t_vals[i], estado, 0.01)
    t_vals[i+1] = t_vals[i] + 0.01
    x_vals[i+1] = estado[0]
    v_vals[i+1] = estado[1]

#
# c) En una misma figura con dos subplots (uno encima del otro):
#      - Arriba: x(t) vs t.  Superponga la solución analítica:
#            x_a(t) = e^{−γt} [cos(ω_d · t) + (γ/ω_d) sin(ω_d · t)]
#        donde  ω_d = √(ω₀² − γ²).
#      - Abajo: v(t) vs t.
#    Guarde como "p3_oscilador.png".

gamma = 0.3
omega_0 = 2 * np.pi
omega_d = np.sqrt(omega_0**2 - gamma**2)

x_analitico = np.exp(-gamma * t_vals) * (np.cos(omega_d * t_vals) + (gamma / omega_d) * np.sin(omega_d * t_vals))

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t_vals, x_vals, label='x(t) numérico')
plt.plot(t_vals, x_analitico, 'r--', label='x(t) analítico')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Posición x (m)')
plt.title('Oscilador Amortiguado: Posición vs Tiempo')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t_vals, v_vals, label='v(t) numérico')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Velocidad v (m/s)')
plt.title('Oscilador Amortiguado: Velocidad vs Tiempo')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#
# d) Imprima el error máximo |x_numérico − x_analítico| en todo el
#    intervalo.
#

error_maximo = np.max(np.abs(x_vals - x_analitico))
print(f"Error máximo |x_numérico − x_analítico|: {error_maximo:.6f}")

# --- Escriba su código aquí ---


# ====================================================================
# PROBLEMA 4 — Ajuste de datos + Análisis de Fourier (25 pts · ~25 min)
# ====================================================================
#
# ── Parte A: Ajuste de decaimiento radiactivo (15 pts) ──
#
# El archivo "datos_decaimiento.csv" contiene mediciones (t, N, σ)
# de la actividad de una muestra radiactiva.
#
# El modelo es:  N(t) = N₀ · e^{−λt}
# Linealizando:  ln(N) = ln(N₀) − λt   →   Y = a₀ + a₁·t
#
# a) Cargue los datos. Calcule Y = ln(N) y la incertidumbre propagada
#    σ_Y = σ / |N|.

datos = np.loadtxt("datos_decaimiento.csv", delimiter=",", skiprows=1)
t = datos[:, 0]
N = datos[:, 1]
sigma = datos[:, 2]

Y = np.log(N)
sigma_Y = sigma / np.abs(N)

#
# b) Construya la matriz normal A (2×2) y el vector b (2×1) para
#    el ajuste lineal ponderado Y = a₀ + a₁·t, y resuelva con
#    minimos_cuadrados. Extraiga N₀ = e^{a₀} y λ = −a₁.
#    Imprima N₀ y λ.

A = np.zeros((2, 2))
b = np.zeros(2)

A[0,0] = np.sum((1 / sigma_Y**2))
A[0,1] = A[1,0] = np.sum((t / sigma_Y**2))
A[1,1] = np.sum((t**2 / sigma_Y**2))

b[0] = np.sum((Y / sigma_Y**2))
b[1] = np.sum((t * Y / sigma_Y**2))

coefs = fc.minimos_cuadrados(A, b)
intercepto = coefs[0]
pendiente = coefs[1]

N0 = np.exp(intercepto)
lambda_ = -pendiente

print(f"N0 = {N0:.6f}")
print(f"λ = {lambda_:.6f}")

#
# c) Calcule χ² y χ²/ν (con ν = N_datos − 2).
#    Imprima ambos valores.

chi2 = np.sum(((Y - (intercepto + pendiente * t)) / sigma_Y)**2)
nu = len(t) - 2
chi2_red = chi2 / nu

print(f"χ² = {chi2:.6f}")
print(f"χ²/ν = {chi2_red:.6f}")


#
# d) Grafique: datos con barras de error + curva ajustada N₀·e^{−λt}.
#    Incluya en la gráfica el valor de N₀, λ y χ²/ν como texto.
#    Guarde como "p4a_decaimiento.png".

t_fit = np.linspace(min(t), max(t), 100)
N_fit = N0 * np.exp(-lambda_ * t_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(t, N, yerr=sigma, fmt='o', label='Datos con error')
plt.plot(t_fit, N_fit, 'r-', label='Ajuste: N₀·e^{−λt}')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Actividad N')
plt.title('Ajuste de Decaimiento Radiactivo')
plt.legend()
plt.text(0.05, 0.95, f'N₀ = {N0:.2f}\nλ = {lambda_:.4f} s⁻¹\nχ²/ν = {chi2_red:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
plt.grid()
plt.show()

#
#
# ── Parte B: Análisis espectral de una señal (10 pts) ──
#
# El archivo "senal_ruido.csv" contiene una señal V(t) muestreada
# uniformemente con dt = 0.01 s (128 puntos).
#
# e) Cargue los datos. Calcule la FFT de la señal V usando fc.fft().

senal = np.loadtxt("senal_ruido.csv", delimiter=",", skiprows=1)
V = senal[:, 1]
dt = 0.01
N = len(V)

FFT = fc.fft(V)

#
# f) Construya el eje de frecuencias:  freq_k = k / (N·dt)  para
#    k = 0, 1, ..., N/2.
#    Grafique |FFT[k]| vs frecuencia (solo la mitad positiva, k=0..N/2).

FFT_magnitude = np.zeros(N//2 + 1)
freq_k = np.zeros(N//2 + 1)

for k in range(N//2 + 1):
    freq_k[k] = k / (N * dt)
    FFT_magnitude[k] = np.abs(FFT[k])
    print(f"Frecuencia k={k}: {freq_k[k]:.2f} Hz")
    print(f"FFT[{k}] = {FFT[k]:.4f} + {FFT[k].imag:.4f}j, |FFT| = {FFT_magnitude[k]:.4f}")

plt.figure(figsize=(10, 6))
plt.stem(freq_k, FFT_magnitude, 'o-')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|FFT|')
plt.title('Espectro de la señal V(t)')
plt.grid()
plt.show()

#
# g) Identifique las dos frecuencias dominantes (los dos picos más altos
#    excluyendo k=0). Imprima sus valores en Hz.
#    Guarde la gráfica como "p4b_espectro.png".
#

maximos = (np.diff(np.sign(np.diff(FFT_magnitude))) < 0).nonzero()[0] + 1
print(f"Índices de máximos locales: {maximos}")
picos = [(k, FFT_magnitude[k]) for k in maximos if k != 0]
picos_ordenados = sorted(picos, key=lambda x: x[1], reverse=True)
frecuencias_dominantes = [freq_k[k] for k, mag in picos_ordenados[:2]]
print(f"Frecuencias dominantes: {frecuencias_dominantes[0]:.2f} Hz, {frecuencias_dominantes[1]:.2f} Hz")

# --- Escriba su código aquí ---

print("\n" + "=" * 50)
print("  FIN DEL EXAMEN — Verifique sus gráficas y salidas")
print("=" * 50)
