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


print("\n" + "=" * 50)
print("  FIN DEL EXAMEN — Verifique sus gráficas y salidas")
print("=" * 50)
