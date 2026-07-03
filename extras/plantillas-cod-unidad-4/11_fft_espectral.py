"""
PLANTILLA 11 - Análisis espectral con FFT (Transformada Rápida de Fourier)
===========================================================================
La FFT descompone una señal en sus componentes de frecuencia.

numpy.fft.fft(u)     → espectro complejo U[k]  (N puntos, señal discreta)
numpy.fft.rfft(u)    → solo la mitad positiva (señal real)  N//2+1 puntos
numpy.fft.fftfreq(N, d=dx)  → frecuencias correspondientes [Hz o 1/m]

Interpretación:
  - |U[k]| / N    → amplitud del modo k
  - angle(U[k])   → fase del modo k
  - k / (N*dx)    → frecuencia espacial f_k [1/m]
  - c * f_k       → frecuencia temporal ν_k [Hz]

Filtrado en frecuencia (filtro pasa-bajos):
    1. Calcular FFT
    2. Poner a cero los coeficientes con |f| > f_corte
    3. Aplicar IFFT para recuperar señal filtrada

Fuente: referencias/cap04_ecuaciones_onda_fluidos.md
Solo usa: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SEÑAL DE PRUEBA: superposición de modos + ruido
# =============================================================================
N   = 512        # número de puntos (potencia de 2 → FFT más eficiente)
dx  = 0.4        # [m] paso espacial
x   = np.arange(N) * dx

# Señal = modo 1 + modo 3 (amplitudes A1, A3) + ruido gaussiano
A1, f1 = 1.0, 0.5    # amplitud y frecuencia espacial [1/m]
A3, f3 = 0.4, 1.5
ruido  = 0.1

u = (A1 * np.sin(2 * np.pi * f1 * x)
     + A3 * np.sin(2 * np.pi * f3 * x)
     + ruido * np.random.randn(N))

# =============================================================================
# FFT Y ESPECTRO DE AMPLITUDES
# =============================================================================
U      = np.fft.fft(u)                  # transformada compleja
freqs  = np.fft.fftfreq(N, d=dx)       # frecuencias [1/m]

# Solo la mitad positiva (señal real → espectro simétrico)
mitad  = N // 2
freqs_pos = freqs[:mitad]
amp_pos   = np.abs(U[:mitad]) / N * 2  # factor 2 por la simetría

print("Picos del espectro (amplitud > 0.05):")
for k, (f, a) in enumerate(zip(freqs_pos, amp_pos)):
    if a > 0.05:
        print(f"  f = {f:.3f} 1/m,  amplitud = {a:.4f}")

# =============================================================================
# FILTRADO PASO-BAJOS (eliminar ruido de alta frecuencia)
# =============================================================================
f_corte = 2.0    # [1/m] frecuencia de corte

U_filtrado = U.copy()
U_filtrado[np.abs(freqs) > f_corte] = 0.0   # eliminar componentes altas

u_filtrado = np.fft.ifft(U_filtrado).real    # transformada inversa

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Panel 1: señal original
ax = axes[0, 0]
ax.plot(x, u, alpha=0.7, label='Original')
ax.plot(x, u_filtrado, linewidth=2, label='Filtrada')
ax.set_xlabel('x [m]')
ax.set_ylabel('u(x)')
ax.set_title('Señal original vs filtrada')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: espectro de amplitudes
ax = axes[0, 1]
ax.stem(freqs_pos, amp_pos, basefmt='k-', markerfmt='C1o')
ax.axvline(f_corte, color='red', linestyle='--', label=f'f_corte = {f_corte}')
ax.set_xlabel('Frecuencia espacial [1/m]')
ax.set_ylabel('Amplitud')
ax.set_title('Espectro de amplitudes |FFT|/N')
ax.legend()
ax.set_xlim(0, 3)
ax.grid(True, alpha=0.3)

# Panel 3: espectro de fase
fase = np.angle(U[:mitad])
ax = axes[1, 0]
ax.plot(freqs_pos, fase, '.', markersize=4)
ax.set_xlabel('Frecuencia espacial [1/m]')
ax.set_ylabel('Fase [rad]')
ax.set_title('Espectro de fase')
ax.set_xlim(0, 3)
ax.grid(True, alpha=0.3)

# Panel 4: espectro de potencia (escala logarítmica)
potencia = np.abs(U[:mitad])**2 / N
ax = axes[1, 1]
ax.semilogy(freqs_pos, potencia + 1e-14)
ax.axvline(f_corte, color='red', linestyle='--', label=f'f_corte = {f_corte}')
ax.set_xlabel('Frecuencia espacial [1/m]')
ax.set_ylabel('Potencia (escala log)')
ax.set_title('Espectro de potencia')
ax.set_xlim(0, 3)
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(f'Análisis espectral FFT  (N={N}, dx={dx} m)', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
# USO TÍPICO PARA ONDA 1D (análisis de modos en la cuerda)
# =============================================================================
# # Después de propagar la onda, aplicar FFT a la instantánea final:
# U_onda = np.fft.rfft(y_actual)
# freqs_onda = np.fft.rfftfreq(Nx, d=dx)
# amp_onda   = np.abs(U_onda) / Nx * 2
# plt.plot(freqs_onda, amp_onda)
# plt.xlabel('Frecuencia espacial [1/m]')
# plt.ylabel('Amplitud')
# plt.title('Modos excitados en la cuerda')
# plt.show()
