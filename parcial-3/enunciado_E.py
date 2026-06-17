"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR — FACULTAD DE CIENCIAS NATURALES          ║
║                    FÍSICA COMPUTACIONAL  —  FCO4101                          ║
║                    PARCIAL III  ·  SIMULACRO  E                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Potencial Eléctrico de un Cuadrupolo en Caja Conductora
  ──────────────────────────────────────────────────────────────────

  Se tiene una caja cuadrada conductora de lado L, aterrizada (V = 0 en todas
  sus paredes). En el interior se colocan cuatro placas cuadradas con densidad
  de carga superficial uniforme, dispuestas en configuración de cuadrupolo:

          ┌──────────────────────────┐
          │   [+ρ₀]      [−ρ₀]      │
          │                          │   V = 0 en todo el borde
          │   [−ρ₀]      [+ρ₀]      │
          └──────────────────────────┘

  El potencial eléctrico V(x,y) satisface la ecuación de Poisson:

              ∇²V(x,y) = −ρ(x,y)/ε₀                              (Poisson)

  y ∇²V = 0 en las regiones sin carga (ecuación de Laplace).

  ──────────────────────────────────────────────────────────────
  DISCRETIZACIÓN: Diferencias Finitas Centrales
  ──────────────────────────────────────────────────────────────

  Con paso de malla Δ uniforme en x e y, la ecuación de Poisson discretizada es:

      V[i,j] = ¼ (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
               + Δ²·ρ[i,j] / (4ε₀)                               (*)

  MÉTODO: Sobrerelajación Sucesiva (SOR)
  ──────────────────────────────────────────────────────────────

  En el método de Gauss-Seidel se usa el valor (*) directamente. En SOR se
  combina con el valor anterior mediante el factor de sobrerelajación ω:

      V[i,j]  ←  (1−ω)·V[i,j]  +  ω·V_GS[i,j]                   (**)

  donde V_GS[i,j] es el valor de Gauss-Seidel dado por (*).
  Para una malla N×N, el valor óptimo teórico es ω_opt ≈ 2/(1 + π/N).
  Se recomienda ω ∈ (1.5, 1.95) en la práctica.

  Condición de convergencia:  max|V^{k+1} − V^k| < tolerancia

  Condiciones de frontera:
      V[0, :] = V[N−1, :] = V[:, 0] = V[:, N−1] = 0   (caja aterrizada)

  Campo eléctrico (diferencias centrales):
      E_x[i,j] = −(V[i+1,j] − V[i−1,j]) / (2Δ)
      E_y[i,j] = −(V[i,j+1] − V[i,j−1]) / (2Δ)

  Parámetros:  N = 100,  Δ = 1,  ε₀ = 1,  ρ₀ = 100,  ω = 1.85,  tol = 10⁻⁵

  Coordenadas de las placas (índices de nodos):
      Placa 1  (+ρ₀): filas [15:35], columnas [15:35]
      Placa 2  (−ρ₀): filas [15:35], columnas [65:85]
      Placa 3  (−ρ₀): filas [65:85], columnas [15:35]
      Placa 4  (+ρ₀): filas [65:85], columnas [65:85]

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Construya la matriz de densidad de carga ρ[i,j] de tamaño N×N.
     Asigne +ρ₀ a las placas 1 y 4 y −ρ₀ a las placas 2 y 3.
     Inicialice V = 0 en toda la malla y aplique las CC de frontera.
     Describa el sistema físico: ¿a qué configuración multipolar corresponde?
     ¿Cuál sería la solución analítica aproximada a gran distancia de las placas?

  2. Implemente la actualización SOR dada por (**). Para cada punto interior
     (i, j) calcule primero V_GS usando (*) y luego aplique la sobrerelajación.
     Explique la diferencia entre Jacobi, Gauss-Seidel y SOR: ¿por qué SOR
     converge más rápido? ¿Qué ocurre si ω > 2 (sobre-sobre-relajación)?

  3. Ejecute el esquema SOR hasta convergencia con tol = 10⁻⁵.
     Registre el error máximo |V^{k+1} − V^k| en cada iteración.
     Grafique el historial de convergencia en escala semilogarítmica.
     ¿Cuántas iteraciones son necesarias? Compare cualitativamente con el
     número esperado para Gauss-Seidel puro (≈ 2×N²/π² pasos extra).

  4. Grafique el potencial V(x,y):
        a) Superficie 3D con colormap 'coolwarm'.
        b) Mapa 2D con líneas equipotenciales superpuestas.
        c) Marque la posición de cada placa en el mapa 2D.
     Describa la forma del potencial: ¿es simétrico? ¿Dónde es máximo/mínimo?
     ¿Por qué V = 0 exactamente en la diagonal que pasa entre placas opuestas?

  5. Calcule el campo eléctrico E = −∇V usando diferencias centrales.
     Grafique las líneas de campo usando quiver (diezmado) o streamplot.
     Superponga las equipotenciales. Describa:
        — ¿Hacia dónde apuntan las líneas de campo cerca de cada placa?
        — ¿Qué ocurre en el punto central de la caja?
        — ¿Cómo se relacionan las líneas de campo con las equipotenciales?

  6. Repita la simulación con ω = 1.0 (Gauss-Seidel puro) y con ω = 1.95.
     Grafique los tres historiales de convergencia en una misma figura.
     Tabule el número de iteraciones para cada ω.
     ¿Qué valor de ω resulta más eficiente para esta malla?
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetros ──────────────────────────────────────────────────────────────
N        = 100
delta    = 1.0
epsilon0 = 1.0
rho0     = 100.0
omega    = 1.85
tol      = 1e-5
N_iter   = 10000

# ─── ÍTEM 1: Densidad de carga y condiciones iniciales ───────────────────────
rho = np.zeros((N, N))

# TODO: asignar +rho0 a placas 1 y 4, -rho0 a placas 2 y 3
# rho[15:35, 15:35] = ...
# rho[15:35, 65:85] = ...
# rho[65:85, 15:35] = ...
# rho[65:85, 65:85] = ...

V = np.zeros((N, N))
# CC: V en bordes = 0 (ya inicializado, no actualizar i=0, i=N-1, j=0, j=N-1)

# ─── ÍTEM 2 y 3: Iteración SOR hasta convergencia ────────────────────────────
historial_error = []

for iteracion in range(N_iter):
    V_old = V.copy()

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # TODO: calcular V_GS (Poisson discretizada)
            # V_GS = 0.25 * (...) + delta**2 * rho[i,j] / (4 * epsilon0)
            # TODO: actualización SOR
            # V[i, j] = (1 - omega) * V[i, j] + omega * V_GS
            pass

    error = np.max(np.abs(V - V_old))
    historial_error.append(error)

    if iteracion % 200 == 0:
        print(f"  Iter {iteracion:5d}   error = {error:.3e}")

    if error < tol:
        print(f"Convergencia en iteración {iteracion}")
        break

# ─── ÍTEM 4: Gráfica del potencial ───────────────────────────────────────────
x = np.arange(N) * delta
y = np.arange(N) * delta
X, Y = np.meshgrid(x, y)

# TODO: superficie 3D + mapa 2D con equipotenciales + posición de placas

# ─── ÍTEM 5: Campo eléctrico ─────────────────────────────────────────────────
Ex = np.zeros_like(V)
Ey = np.zeros_like(V)

# TODO: Ex[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2*delta)
# TODO: Ey[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2*delta)
# TODO: graficar quiver/streamplot superpuesto con equipotenciales

# ─── ÍTEM 6: Comparación de ω ────────────────────────────────────────────────
# TODO: repetir con omega = 1.0 y omega = 1.95, graficar historiales

plt.show()
