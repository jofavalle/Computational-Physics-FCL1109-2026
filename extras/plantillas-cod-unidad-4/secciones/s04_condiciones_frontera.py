"""
SECCIÓN: CONDICIONES DE FRONTERA
=================================
Receta de todos los patrones de CC que aparecen en la unidad 4.
Se aplican DENTRO del loop o en función aparte (patrón N-S).

REGLA: las CC se aplican DESPUÉS de calcular los nodos interiores,
       sobreescribiendo los valores de los bordes.
"""

import numpy as np

# =============================================================================
# CAMPO 1D - BORDES DE UN VECTOR u[0..N-1]
# =============================================================================

# --- Dirichlet: valor fijo (cuerda fija, temperatura fija) ---
u[0]  = 0.0
u[-1] = 0.0

# --- Neumann: derivada cero (extremo libre, aislado) ---
u[0]  = u[1]       # du/dx = 0 en x=0
u[-1] = u[-2]      # du/dx = 0 en x=L

# --- Absorción / salida (extrapolación, N-S) ---
u[-1] = u[-2]      # igual que Neumann pero en una sola dirección

# --- Periódica ---
u[0]  = u[-2]      # u[0] = u[N-2]  (el penúltimo, no el último)
u[-1] = u[1]       # u[N-1] = u[1]


# =============================================================================
# CAMPO 2D - BORDES DE UNA MATRIZ u[Nx, Ny]
# =============================================================================

# --- Dirichlet: todos los bordes a cero ---
u[0,  :] = 0.0    # borde izquierdo   (fila 0)
u[-1, :] = 0.0    # borde derecho     (fila N-1)
u[:,  0] = 0.0    # borde inferior    (columna 0)
u[:, -1] = 0.0    # borde superior    (columna N-1)

# --- Dirichlet: una pared a valor V₀ (Laplace, capacitor) ---
u[0, :]  = 100.0   # borde superior a 100 V
u[-1, :] = 0.0
u[:,  0] = 0.0
u[:, -1] = 0.0

# --- Neumann: borde libre (membrana, clase_28-04-26) ---
u_next[0,  :] = u_next[1,  :]    # ∂u/∂x = 0 en x=0
u_next[-1, :] = u_next[-2, :]    # ∂u/∂x = 0 en x=L
u_next[:,  0] = u_next[:,  1]    # ∂u/∂y = 0 en y=0
u_next[:, -1] = u_next[:, -2]    # ∂u/∂y = 0 en y=L


# =============================================================================
# PATRÓN N-S: CC como FUNCIÓN (se llama antes y después del loop SOR)
# =============================================================================
def aplicar_frontera(psi, w, Nx, Ny, h, V0):
    """
    Condiciones de frontera para Navier-Stokes (lid-driven cavity).
    - Entrada (x=0):  flujo uniforme, ψ linealmente creciente en y
    - Salida (x=Nx):  extrapolación (∂/∂x = 0)
    - Tapa (y=Ny):    placa deslizante a velocidad V0
    - Pared (y=0):    no deslizamiento (ψ = 0)
    """
    # Entrada: perfil lineal en y
    for j in range(Ny):
        psi[0, j] = V0 * j

    # Salida: extrapolación
    psi[-1, :] = psi[-2, :]
    w[-1,   :] = w[-2,   :]

    # Tapa superior: ψ[i, Ny] = ψ[i, Ny-1] + h*V0
    for i in range(1, Nx):
        psi[i, Ny] = psi[i, Ny - 1] + h * V0
        w[i,   Ny] = 0.0

    # Pared inferior: no deslizamiento
    psi[:, 0] = 0.0

    return psi, w


# =============================================================================
# PATRÓN: MÁSCARA para obstáculos interiores (viga, placas del capacitor)
# =============================================================================

# --- Con máscara booleana ---
mascara = np.zeros((Nx, Ny), dtype=bool)
mascara[x0:x1, y0:y1] = True    # región de la viga / placa

# Dentro del loop, saltar si está en la máscara:
# if mascara[i, j]:
#     continue

# --- Con máscara flotante (1.0 = placa fija, 0.0 = libre) ---
mask = np.zeros((N, N))
mask[30, 30:70] = 1    # placa superior del capacitor
mask[60, 30:70] = 1    # placa inferior

# Dentro del loop:
# if mask[i, j] == 0:
#     V[i, j] = 0.25*(V[i+1,j] + ...)   # solo actualizar nodos libres


# =============================================================================
# PATRÓN: vorticidad en la pared (condición de Thom)
# Para N-S: en las paredes sólidas, ω = -2*(ψ_interior - ψ_pared) / h²
# =============================================================================
# Pared inferior (j=0):
w[:, 0] = -2 * (psi[:, 1] - psi[:, 0]) / h**2

# Pared superior (j=Ny):
w[:, -1] = -2 * (psi[:, -2] - psi[:, -1]) / h**2

# Pared izquierda (i=0):
w[0, :] = -2 * (psi[1, :] - psi[0, :]) / h**2
