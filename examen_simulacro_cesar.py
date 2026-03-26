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


# --- Escriba su código aquí ---
#Parametros
eps = 0.0103
sig = 3.40
#a)
def V(r):
    return 4*eps*((sig/r)**12 - (sig/r)**6)
#b)
def dV(r):
    return fc.derivada_enesima(V, r, 1)

r_eq = fc.newton_raphson(dV, 3.5, 1e-6, 1e-6, 1000)
print(f"El punto donde la fuerza F(r) = 0 es: {r_eq:.6f} A")
#c)
dV2 = fc.derivada_enesima(V, r_eq, 2)
print(f"La constante de resorte efectiva es: {dV2:.6f} eV/A**2") 
#d)
r_values = np.linspace(3.0, 7.0, 100)

r_min = V(r_eq)

plt.figure(figsize=(8, 5))
plt.plot(r_values, V(r_values), "b-", linewidth=1.5, label="$V(r)$")
plt.plot(r_eq, r_min, "ro")
plt.xlabel("Distancia equilibrio r_eq", fontsize=13)
plt.ylabel("Potencial V(r_eq)", fontsize=13)
plt.title("r_eq vs. V(r_eq)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
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
#
# b) Calcule la rapidez más probable v_p (donde f(v) es máxima)
#    usando Newton-Raphson sobre f'(v) = 0.
#    (Sugerencia: x₀ = 300 m/s.)
#    Imprima v_p y compárelo con el valor analítico v_p = √(2kT/m).
#
# c) Calcule la probabilidad de que una partícula tenga rapidez
#    v > 600 m/s, es decir P = ∫₆₀₀^∞ f(v) dv.
#    Imprima P.
#
# d) Grafique f(v) en el rango v ∈ [0, 1000] m/s.
#    Sombree (con plt.fill_between) la región v > 600 m/s.
#    Marque v_p con una línea vertical punteada.
#    Guarde como "p2_maxwell.png".
#


# --- Escriba su código aquí ---
#a)
m = 6.63e-26
T = 300
k_B = 1.381e-23
def f(v):
    return 4 * np.pi * (m / (2 * np.pi * k_B * T))**(3/2) * v**2 * np.exp((-m * v**2) / (2 * k_B * T))
norm = fc.integral_impropia(f, 0, np.inf, 100000)
print(f"La integral de normalización es: {norm:.6f}")

#b)
def df(v):
    return fc.derivada_enesima(f,v, 1)
v_p = fc.newton_raphson(df, 300, 1e-6, 1e-6, 1000)
v_pn = np.sqrt((2 * k_B * T)/m)
print(f"El valor númerico de la velocidad más probable es: {v_p:.6f} m/s, mientras que el valor análitico es {v_pn:.6f} m/s")

#c)
P = fc.integral_impropia(f, 600, np.inf)
print(f"La probabilidad de que una partícula tenga rapidez v > 600 es: {P}")

#d)
v_vals = np.linspace(0, 1000, 100)
plt.figure(figsize=(8, 5))
plt.plot(v_vals, f(v_vals), "b-", linewidth=1.5, label="$f(v)$")
plt.fill_between(v_vals, f(v_vals), where = (v_vals > 600))
plt.axvline(v_p, label = "v_p", linestyle = "--")
plt.xlabel("Velocidad promedio", fontsize=13)
plt.ylabel("Probabilidad", fontsize=13)
plt.title("P vs. V", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
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
#
# b) Integre desde t = 0 hasta t = 15 s usando rk4 con dt = 0.01.
#    Almacene los arreglos de t, x(t) y v(t).
#
# c) En una misma figura con dos subplots (uno encima del otro):
#      - Arriba: x(t) vs t.  Superponga la solución analítica:
#            x_a(t) = e^{−γt} [cos(ω_d · t) + (γ/ω_d) sin(ω_d · t)]
#        donde  ω_d = √(ω₀² − γ²).
#      - Abajo: v(t) vs t.
#    Guarde como "p3_oscilador.png".
#
# d) Imprima el error máximo |x_numérico − x_analítico| en todo el
#    intervalo.
#


# --- Escriba su código aquí ---
#a)
w0 = 2*np.pi
gamma = 0.3
# N pasos = diferencia entre tiempo / dt
# N points = N pasos + 1
N_pasos = int((15-0)/0.01)
# Condiciones iniciales
x0 = 1
v0 = 0
estado = [x0, v0]
def f(t, estado):
    dxdt = estado[1]
    dvdt = -2*gamma*estado[1] - w0**2 * estado[0]
    return np.array([dxdt, dvdt], float)

#b)
x_points = np.zeros(N_pasos+1)
v_points = np.zeros(N_pasos+1)
t_points = np.linspace(0, 15, N_pasos+1)

for i in range(N_pasos):
    x_points[i] = estado[0]
    v_points[i] = estado[1]

    estado = fc.rk4(f, t_points[i], estado, 0.01)
#c)
w_d = np.sqrt(w0**2 - gamma**2)
f_an = np.exp(-gamma*t_points) * (np.cos(w_d * t_points) + (gamma/w_d)*np.sin(w_d*t_points))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(t_points, x_points, "b-", linewidth=1.5, label = "Posición con RK4")
ax1.plot(t_points, f_an, color = "orange", linewidth=1.5, label = "Posición analitica")
ax1.set_xlabel("$t$ [s]", fontsize=13)
ax1.set_ylabel("$x$ [m]", fontsize=13)
ax1.set_title("Posición", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_points, v_points, "r-", linewidth=1.5, label = "Velocidad con RK4")
ax2.set_xlabel("$t$ [s]", fontsize=13)
ax2.set_ylabel("$v$ [m/s]", fontsize=13)
ax2.set_title("Velocidad", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Oscilador armónico amortiguado", fontsize=15)
fig.tight_layout()
plt.show()

print(f"El error máximo cometido es {np.max(x_points - f_an)}")

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
#
# b) Construya la matriz normal A (2×2) y el vector b (2×1) para
#    el ajuste lineal ponderado Y = a₀ + a₁·t, y resuelva con
#    minimos_cuadrados. Extraiga N₀ = e^{a₀} y λ = −a₁.
#    Imprima N₀ y λ.
#
# c) Calcule χ² y χ²/ν (con ν = N_datos − 2).
#    Imprima ambos valores.
#
# d) Grafique: datos con barras de error + curva ajustada N₀·e^{−λt}.
#    Incluya en la gráfica el valor de N₀, λ y χ²/ν como texto.
#    Guarde como "p4a_decaimiento.png".
#
#
# ── Parte B: Análisis espectral de una señal (10 pts) ──
#
# El archivo "senal_ruido.csv" contiene una señal V(t) muestreada
# uniformemente con dt = 0.01 s (128 puntos).
#
# e) Cargue los datos. Calcule la FFT de la señal V usando fc.fft().
#
# f) Construya el eje de frecuencias:  freq_k = k / (N·dt)  para
#    k = 0, 1, ..., N/2.
#    Grafique |FFT[k]| vs frecuencia (solo la mitad positiva, k=0..N/2).
#
# g) Identifique las dos frecuencias dominantes (los dos picos más altos
#    excluyendo k=0). Imprima sus valores en Hz.
#    Guarde la gráfica como "p4b_espectro.png".
#


# --- Escriba su código aquí ---
#Parte A
#a)
datos = np.loadtxt("datos_decaimiento.csv", delimiter=",", skiprows=1)

t = datos[:, 0]       # primera columna
N = datos[:, 1]       # segunda columna
sigma = datos[:, 2]   # tercera columna

Y = np.log(N)
sigy = sigma/abs(N)
#b)
A = np.zeros((2,2))
b = np.zeros(2)

A[0,0] = np.sum(1/sigy**2)
A[0,1] = A[1,0] = np.sum(t/sigy**2)
A[1,1] = np.sum(t**2/sigy**2)

b[0] = np.sum(Y/sigy**2)
b[1] = np.sum(Y*t/sigy**2)

coef = fc.minimos_cuadrados(A, b)
print(f"El número de núcleos iniciales es: {np.exp(coef[0])}, con una constante de decaimiento {coef[1]}")
#A = np.array([[A00, A01], [A10, A11]])
#b = np.array([b0, b1])

#c)
chi2 = fc.chi_cuadrada(Y,(coef[0]+coef[1]*t),sigy)
chindof = chi2/(len(N)-2)
print(f"El chi**2 es: {chi2} y el chino rmalizado {chindof}")
#d)
N0 = (np.exp(coef[0]))
plt.figure(figsize=(8, 5))
plt.errorbar(t, N, yerr = sigma, fmt='o', label="LA DATA")
plt.plot(t,N0*np.exp(t*coef[1]),label="Ajuste" )
plt.xlabel("tiempo", fontsize=13)
plt.ylabel("Nucleos", fontsize=13)
plt.title("N vs. t", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Parte B
#e) 
señal = np.loadtxt("senal_ruido.csv", delimiter=",", skiprows=1)

t = señal[:, 0]       # primera columna
V = señal[:, 1]       # segunda columna

fourier = fc.fft(V)
dt = 0.01
k = 0.0
N = len(V)

freq_k = np.zeros(N//2 + 1)
fourier_mag = np.zeros(N//2 +1)

#f)
for k in range(N//2+1):
    freq_k[k] = k/(N*dt)
    fourier_mag[k] = np.abs(fourier[k])
    print(f"Frecuencia k={k}: {freq_k[k]:.2f} Hz")
    print(f"FFT[{k}] = {fourier[k]:.4f} + {fourier[k].imag:.4f}j, |FFT| = {fourier_mag[k]:.4f}")

plt.figure(figsize=(8, 5))
plt.stem(freq_k, fourier_mag, "o-")
plt.xlabel("Frecuencia (Hz)", fontsize=13)
plt.ylabel("FFT", fontsize=13)
plt.title("FFT vs. Frecuencia", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#g)
indices = (np.diff(np.sign(np.diff(fourier_mag))) < 0).nonzero()[0] + 1
print(f"Índices de máximos locales: {indices}")
picos = [(k, fourier_mag[k]) for k in indices if k != 0]
picos_ordenados = sorted(picos, key=lambda x: x[1], reverse=True)
frecuencias_dominantes = [freq_k[k] for k, mag in picos_ordenados[:2]]
print(f"Frecuencias dominantes: {frecuencias_dominantes[0]:.2f} Hz, {frecuencias_dominantes[1]:.2f} Hz")

print("\n" + "=" * 50)
print("  FIN DEL EXAMEN — Verifique sus gráficas y salidas")
print("=" * 50)
