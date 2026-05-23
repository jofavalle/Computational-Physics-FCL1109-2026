"""
PLANTILLA 02 — Modos normales de la cuerda vibrante
====================================================
Solución analítica:
    y(x,t) = Σ Bₙ sin(nπx/L) cos(nπct/L)

Los coeficientes Bₙ se obtienen de la condición inicial y(x,0) por
proyección ortogonal sobre los modos seno:
    Bₙ = (2/L) ∫₀ᴸ y(x,0) sin(nπx/L) dx   ≈ suma trapezoidal

Frecuencias de los modos:
    fₙ = n*c / (2*L)   [Hz]
    ωₙ = n*π*c / L     [rad/s]

Análisis espectral (FFT):
    La FFT de la señal temporal en un punto x₀ da los modos excitados.

Fuente: clase_16-04-26.ipynb
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L      = 1.0    # [m]   longitud de la cuerda
c      = 1.0    # [m/s] velocidad de propagación
N_modo = 20     # número de modos a incluir en la reconstrucción

# =============================================================================
# CONDICIÓN INICIAL y(x,0)
# =============================================================================
Nx = 200
x  = np.linspace(0, L, Nx)

# --- Opción A: pulso gaussiano ---
sigma = L / 10
x0    = L / 2
y_inicial = np.exp(-((x - x0)**2) / (2 * sigma**2))

# --- Opción B: cuerda triangular (descomentar) ---
# y_inicial = np.where(x < L/2, 2*x/L, 2*(L-x)/L)

# =============================================================================
# CÁLCULO DE COEFICIENTES Bₙ (proyección trapezoidal)
# =============================================================================
B = np.zeros(N_modo + 1)    # B[n] = coeficiente del modo n

for n in range(1, N_modo + 1):
    integrando = y_inicial * np.sin(n * np.pi * x / L)
    # regla del trapecio manual:
    B[n] = (2.0 / L) * np.trapezoid(integrando, x)

print("Coeficientes Bₙ (primeros 10):")
for n in range(1, min(11, N_modo + 1)):
    print(f"  B[{n:2d}] = {B[n]: .6f}")

# =============================================================================
# RECONSTRUCCIÓN TEMPORAL DE y(x,t)
# =============================================================================
Nt     = 400
t_max  = 2 * L / c            # un período de oscilación del modo fundamental
t_vals = np.linspace(0, t_max, Nt)

# Instantáneas en ciertos tiempos
tiempos_plot = [0, Nt//4, Nt//2, 3*Nt//4]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for idx, ti in enumerate(tiempos_plot):
    t = t_vals[ti]
    ax = axes.flat[idx]

    y_rec = np.zeros(Nx)
    for n in range(1, N_modo + 1):
        omega_n = n * np.pi * c / L
        y_rec  += B[n] * np.sin(n * np.pi * x / L) * np.cos(omega_n * t)

    ax.plot(x, y_rec)
    ax.set_title(f't = {t:.3f} s')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

fig.suptitle(f'Reconstrucción con {N_modo} modos normales', fontsize=13)
plt.tight_layout()
plt.show()

# =============================================================================
# ESPECTRO DE AMPLITUDES |Bₙ| vs FRECUENCIA fₙ
# =============================================================================
frecuencias = np.array([n * c / (2 * L) for n in range(1, N_modo + 1)])
amplitudes  = np.abs(B[1:N_modo + 1])

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.stem(frecuencias, amplitudes, basefmt='k-')
ax2.set_xlabel('Frecuencia fₙ [Hz]')
ax2.set_ylabel('|Bₙ|')
ax2.set_title('Espectro de modos normales')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# ANÁLISIS ESPECTRAL CON FFT (señal temporal en x = L/2)
# =============================================================================
# Evaluamos y(L/2, t) para todos los tiempos
y_centro = np.zeros(Nt)
for ti, t in enumerate(t_vals):
    for n in range(1, N_modo + 1):
        omega_n    = n * np.pi * c / L
        y_centro[ti] += B[n] * np.sin(n * np.pi * (L/2) / L) * np.cos(omega_n * t)

# FFT de la señal temporal
dt_fft   = t_vals[1] - t_vals[0]
Y_fft    = np.fft.rfft(y_centro)
freqs    = np.fft.rfftfreq(Nt, d=dt_fft)
amp_fft  = np.abs(Y_fft) / Nt

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(freqs, amp_fft)
ax3.set_xlabel('Frecuencia [Hz]')
ax3.set_ylabel('Amplitud')
ax3.set_title('Espectro FFT de y(L/2, t)')
ax3.set_xlim(0, (N_modo + 2) * c / (2 * L))
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
