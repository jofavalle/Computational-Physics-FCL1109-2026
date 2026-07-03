"""
SECCIÓN: POSTPROCESAMIENTO
===========================
Operaciones que se hacen DESPUÉS del loop de integración:
cálculos derivados del campo resuelto (velocidades, energía, campo E).

Aparece en todos los scripts de N-S y Laplace/Poisson.
"""

import numpy as np

# =============================================================================
# 1. VELOCIDADES A PARTIR DE LA FUNCIÓN DE CORRIENTE ψ (N-S)
# =============================================================================
# Relaciones:  vx = ∂ψ/∂y   ,   vy = -∂ψ/∂x
# Diferencias finitas centradas: f'(x) ≈ (f[i+1] - f[i-1]) / (2h)

vx = np.zeros_like(psi)
vy = np.zeros_like(psi)

# Vectorizado (más rápido)
vx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)    # ∂ψ/∂y
vy[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * h)   # -∂ψ/∂x

# Con loop (más explícito)
for i in range(1, Nx):
    for j in range(1, Ny):
        vx[i, j] =  (psi[i, j+1] - psi[i, j-1]) / (2 * h)
        vy[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * h)


# =============================================================================
# 2. CAMPO ELÉCTRICO A PARTIR DEL POTENCIAL V (Laplace/Poisson)
# =============================================================================
# Relación: E = -∇V   →   Ex = -∂V/∂x,   Ey = -∂V/∂y

Ex = np.zeros_like(V)
Ey = np.zeros_like(V)

# Diferencias finitas centradas (solo nodos interiores)
Ex[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dx)    # -∂V/∂x
Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dx)    # -∂V/∂y

# Magnitud del campo
E_mag = np.sqrt(Ex**2 + Ey**2)


# =============================================================================
# 3. ENERGÍA MECÁNICA DE UNA CUERDA/MEMBRANA
# =============================================================================
# E_cin = 0.5 * ρ * (∂y/∂t)²    → aproximar ∂y/∂t ≈ (y_actual - y_anterior)/dt
# E_pot = 0.5 * T * (∂y/∂x)²

v_campo = (u_actual - u_anterior) / dt                # velocidad del campo

E_cin = 0.5 * rho * v_campo**2                        # energía cinética densidad
E_pot = 0.5 * T * np.gradient(u_actual, dx)**2        # energía potencial densidad
E_total = np.sum(E_cin + E_pot) * dx                  # integrar en x


# =============================================================================
# 4. ANÁLISIS ESPECTRAL (FFT de la instantánea final)
# =============================================================================
U     = np.fft.rfft(u_actual)              # FFT de señal real (N//2+1 puntos)
freqs = np.fft.rfftfreq(len(u_actual), d=dx)  # frecuencias espaciales [1/m]
amp   = np.abs(U) / len(u_actual) * 2     # amplitudes (factor 2 por simetría)

# Frecuencias de modos normales (para comparar):
# f_n = n*c / (2*L)


# =============================================================================
# 5. MÁSCARA PARA EXCLUIR REGIÓN SÓLIDA EN LA VISUALIZACIÓN (N-S con viga)
# =============================================================================
mascara_viga = np.zeros((Nx, Ny), dtype=bool)
mascara_viga[x0:x1, y0:y1] = True

# Reemplazar por NaN para que imshow/contourf no grafique la región
psi_vis = np.where(mascara_viga, np.nan, psi)
w_vis   = np.where(mascara_viga, np.nan, w)


# =============================================================================
# 6. NORMA DEL ERROR (convergencia SOR o comparación analítica)
# =============================================================================
# Error máximo (norma infinita): cuánto cambió la solución en la última iteración
error_max = np.max(np.abs(V - V_old))

# Error cuadrático medio respecto a solución analítica
error_rms = np.sqrt(np.mean((V - V_analitica)**2))

# Verificar convergencia en un loop:
if error_max < tolerancia:
    print(f"Convergió en {it + 1} iteraciones  (error = {error_max:.2e})")
    break


# =============================================================================
# 7. VELOCIDAD LOCAL v(x) = sqrt(T(x)/rho(x)) - onda con propiedades variables
# =============================================================================
v_local = np.sqrt(T / rho)
vmax    = np.max(v_local)
dt      = 0.4 * dx / vmax      # paso temporal conservador


# =============================================================================
# 8. COEFICIENTES DE MODOS NORMALES Bₙ
# =============================================================================
L  = 1.0
Nx = 200
x  = np.linspace(0, L, Nx)
N_modos = 20

B = np.zeros(N_modos + 1)
for n in range(1, N_modos + 1):
    integrando = y_inicial * np.sin(n * np.pi * x / L)
    B[n] = (2.0 / L) * np.trapz(integrando, x)    # regla del trapecio

# np.trapz(y, x)  →  ∫ y dx  (método del trapecio)
# np.trapezoid(y, x)  →  nombre nuevo en numpy >= 2.0 (misma función)
