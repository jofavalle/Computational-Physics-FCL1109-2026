"""
SECCIÓN: CONDICIONES INICIALES
================================
Receta de todas las condiciones iniciales que aparecen en los scripts
de la unidad 4. Usadas para campos 1D y 2D.

La mayoría de los problemas arranca con t=0 y t=1 iguales (velocidad cero),
o con un estado calculado analíticamente.
"""

import numpy as np

# Se asume que ya existe: x, Nx, dx, L
# ============================================================

# =============================================================================
# CAMPO 1D
# =============================================================================

# --- Pulso gaussiano (el más común en clase) ---
sigma = L / 10
x0    = L / 2
u = np.exp(-((x - x0)**2) / (2 * sigma**2))

# --- Pulso gaussiano angosto (clase_21-04-26) ---
u = np.exp(-200 * (x - 0.5)**2)

# --- Pulso gaussiano centrado en x=80 (clase_19-05-26, FDTD) ---
u = np.exp(-((x - 80) / 15)**2)

# --- Modo normal n (seno puro, excita solo ese modo) ---
n = 1
u = np.sin(n * np.pi * x / L)

# --- Cuerda triangular (pulsada en el centro) ---
u = np.where(x < L / 2, 2 * x / L, 2 * (L - x) / L)

# --- Rectangular (escalón) ---
u = np.zeros_like(x)
u[60:100] = 1.0

# --- Escalón suavizado con tanh (Burgers) ---
u = 0.5 * (1 - np.tanh(x / 5 - 5))

# --- Ruido aleatorio (Burgers, análisis de estabilidad) ---
u = 0.2 * np.random.rand(len(x))

# --- Pulso sinusoidal modulado (clase_21-05-26, FDTD) ---
longitud_onda = 30.0
k   = 2 * np.pi / longitud_onda
phi = 0.0
envolvente = np.exp(-((x - 80) / 15)**2)
u = envolvente * np.sin(k * x + phi)


# =============================================================================
# CAMPO 2D — construido sobre meshgrid
# =============================================================================
# Se asume que ya existe: X, Y de meshgrid (indexing='ij')

# --- Pulso gaussiano 2D centrado ---
u = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# --- Modo normal 2D (membrana) ---
u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

# --- Ring soliton (clase_28-04-26) ---
R_radio = np.sqrt(X**2 + Y**2)
u = 4 * np.arctan(np.exp(3 - R_radio))


# =============================================================================
# INICIALIZAR DOS PASOS TEMPORALES (leapfrog / onda)
# =============================================================================
# El esquema de onda necesita u[n] y u[n-1]:
u_actual   = u.copy()
u_anterior = u.copy()   # velocidad inicial = 0  →  u[-1] = u[0]
u_nuevo    = np.zeros_like(u)

# Si hay velocidad inicial v0 ≠ 0:
# u_anterior = u_actual - dt * v0


# =============================================================================
# NOTAS RÁPIDAS
# =============================================================================
# np.copy(u)    → copia profunda, u_copia es independiente
# u.copy()      → equivalente, más idiomático
# u_old[:] = u  → sobrescribir sin crear nuevo objeto (más eficiente en loops)
# np.zeros_like(u) → arreglo de ceros con la misma forma y dtype
