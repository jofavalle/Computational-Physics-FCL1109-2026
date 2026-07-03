"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR — FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  —  FCO4101                            ║
║                    PARCIAL III  ·  SIMULACRO  E                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Cuadrupolo de Placas + Electrodo Central a Potencial Fijo
  ──────────────────────────────────────────────────────────────────

  Se tiene una caja cuadrada conductora de lado L, aterrizada (V = 0 en todas
  sus paredes).  En su interior coexisten DOS tipos de fuente:

    (A) Cuatro placas cuadradas con densidad de carga superficial uniforme,
        dispuestas en configuración de CUADRUPOLO:

          ┌──────────────────────────┐
          │   [+ρ₀]        [−ρ₀]    │
          │          ╭───╮           │   V = 0 en todo el borde (tierra)
          │          │+V_e│          │   ← electrodo central conductor
          │          ╰───╯           │
          │   [−ρ₀]        [+ρ₀]    │
          └──────────────────────────┘

    (B) Un ELECTRODO circular conductor en el centro, de radio R_e, mantenido
        a un potencial FIJO V_e = 500 V (p.ej. conectado a una batería).

  Esta es la situación típica de un CAPACITOR / problema de electrostática con
  conductores: parte de la frontera tiene carga prescrita (las placas, vía ρ)
  y parte tiene POTENCIAL prescrito (la caja a 0 V y el electrodo a 500 V).
  El electrodo y la caja aterrizada forman, de hecho, las dos "armaduras" de un
  condensador cuya región intermedia contiene además el cuadrupolo de carga.

  El potencial V(x,y) satisface la ecuación de Poisson:

              ∇²V(x,y) = −ρ(x,y)/ε₀                              (Poisson)

  con ∇²V = 0 donde no hay carga (Laplace), SUJETA a las condiciones de
  contorno de Dirichlet:

        V = 0     en las paredes de la caja
        V = V_e   en toda la superficie (y el interior) del electrodo central

  ──────────────────────────────────────────────────────────────
  DISCRETIZACIÓN: Diferencias Finitas Centrales
  ──────────────────────────────────────────────────────────────

  Con paso de malla Δ uniforme en x e y, la ecuación de Poisson discretizada es:

      V[i,j] = ¼ (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
               + Δ²·ρ[i,j] / (4ε₀)                               (*)

  Los nodos donde V está prescrito (paredes y electrodo) NO se actualizan: se
  mantienen fijos en su valor de Dirichlet durante toda la iteración.

  MÉTODO: Sobrerelajación Sucesiva (SOR)
  ──────────────────────────────────────────────────────────────

  Gauss-Seidel usa el valor (*) directamente.  SOR lo combina con el valor
  anterior mediante el factor de sobrerelajación ω:

      V[i,j]  ←  (1−ω)·V[i,j]  +  ω·V_GS[i,j]                   (**)

  Para una malla N×N, el valor óptimo teórico es ω_opt ≈ 2/(1 + π/N).
  Se recomienda ω ∈ (1.5, 1.95).  Si ω ≥ 2 el método DIVERGE.

  Condición de convergencia:  max|V^{k+1} − V^k| < tol

  Campo eléctrico (diferencias centrales):
      E_x[i,j] = −(V[i,j+1] − V[i,j−1]) / (2Δ)
      E_y[i,j] = −(V[i+1,j] − V[i−1,j]) / (2Δ)

  Parámetros:  N = 100,  Δ = 1,  ε₀ = 1,  ρ₀ = 10,  ω = 1.85,  tol = 10⁻⁵

  Geometría (índices de nodos):
      Placa 1  (+ρ₀): filas [15:35], columnas [15:35]
      Placa 2  (−ρ₀): filas [15:35], columnas [65:85]
      Placa 3  (−ρ₀): filas [65:85], columnas [15:35]
      Placa 4  (+ρ₀): filas [65:85], columnas [65:85]
      Electrodo central (V_e = 500): círculo de radio R_e = 8 centrado en (50,50)

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Construya la matriz de densidad de carga ρ[i,j] (cuadrupolo) y la MÁSCARA
     booleana del electrodo central.  Inicialice V = 0 e imponga las dos CC de
     Dirichlet: paredes a 0 V y electrodo a V_e = 500 V.
     Describa el sistema físico: ¿qué papel juega el electrodo central frente a
     la caja aterrizada (analogía de capacitor)?  Argumente, usando la linealidad
     de Poisson, que V = V_cuadrupolo + V_capacitor (descomposición por
     superposición de la parte antisimétrica y la parte simétrica).

  2. Implemente la actualización SOR (**).  CLAVE: los nodos prescritos (paredes
     y electrodo) deben permanecer FIJOS — nunca se actualizan.  Explique la
     diferencia entre Jacobi, Gauss-Seidel y SOR, por qué SOR converge más
     rápido, y qué ocurre si ω ≥ 2.  ¿Por qué fijar el electrodo equivale a la
     condición física de un conductor a potencial constante (E = 0 en su seno)?

  3. Ejecute SOR hasta convergencia con tol = 10⁻⁵, registrando el error máximo
     |V^{k+1} − V^k| por iteración.  Grafique el historial en escala semilog.
     ¿Cuántas iteraciones se necesitan?  Compárelas con Gauss-Seidel puro.

  4. Grafique el potencial V(x,y):
        a) Superficie 3D (colormap 'coolwarm').
        b) Mapa 2D con líneas equipotenciales.
        c) Marque las cuatro placas y el electrodo central.
     ¿Es el potencial simétrico?  ¿Sigue siendo V = 0 en las diagonales como en
     el cuadrupolo puro, o el electrodo lo modifica?  ¿Dónde es máximo y mínimo?

  5. Calcule el campo eléctrico E = −∇V (diferencias centrales) y grafique sus
     líneas (quiver diezmado o streamplot) sobre las equipotenciales.  Describa:
        — ¿Hacia dónde apuntan las líneas cerca del electrodo (+500 V) y cerca
          de cada placa?
        — ¿Cuánto vale E dentro del electrodo?  ¿Por qué?
        — Relación entre líneas de campo y equipotenciales.

  6. (Capacitor) Estime la carga total sobre el electrodo central aplicando la
     ley de Gauss en 2D sobre un lazo rectangular que lo rodee (sin encerrar las
     placas):
            Q_e = ε₀ ∮ E·n̂ dl
     A partir de ella estime la "capacitancia" C = Q_e / V_e entre el electrodo
     y la caja aterrizada.  Comente el resultado.

  7. Repita la simulación con ω = 1.0 (Gauss-Seidel puro) y ω = 1.95.
     Grafique los tres historiales de convergencia juntos y tabule el número de
     iteraciones.  ¿Qué ω es más eficiente?  Compárelo con ω_opt ≈ 2/(1+π/N).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetros ──────────────────────────────────────────────────────────────
N        = 100
delta    = 1.0
epsilon0 = 1.0
rho0     = 10.0
omega    = 1.85
tol      = 1e-5
N_iter   = 10000

# Electrodo central a potencial fijo
cx = cy  = N // 2     # centro de la caja
R_elec   = 8          # [nodos] radio del electrodo
V_elec   = 500.0      # [V] potencial fijo del electrodo

# ─── ÍTEM 1: Densidad de carga, electrodo y condiciones iniciales ────────────
rho = np.zeros((N, N))
# TODO: asignar +rho0 a placas 1 y 4, -rho0 a placas 2 y 3

# TODO: máscara booleana del electrodo circular (radio R_elec, centro (cx,cy))
# TODO: V = 0 en toda la malla; V[electrodo] = V_elec; bordes a 0 (Dirichlet)

# ─── ÍTEM 2 y 3: Iteración SOR (sin actualizar nodos fijos) ──────────────────
historial_error = []

# TODO: bucle SOR que NO toque paredes ni electrodo; registrar el error

# ─── ÍTEM 4: Gráfica del potencial ───────────────────────────────────────────
x = np.arange(N) * delta
y = np.arange(N) * delta
X, Y = np.meshgrid(x, y)
# TODO: superficie 3D + mapa 2D con equipotenciales + placas + electrodo

# ─── ÍTEM 5: Campo eléctrico ─────────────────────────────────────────────────
# TODO: Ex, Ey por diferencias centrales; quiver/streamplot + equipotenciales

# ─── ÍTEM 6: Carga y capacitancia del electrodo (ley de Gauss 2D) ────────────
# TODO: flujo de E sobre un lazo que rodee el electrodo → Q_e → C = Q_e/V_elec

# ─── ÍTEM 7: Comparación de ω ────────────────────────────────────────────────
# TODO: repetir con omega = 1.0 y 1.95; graficar los tres historiales

plt.show()
