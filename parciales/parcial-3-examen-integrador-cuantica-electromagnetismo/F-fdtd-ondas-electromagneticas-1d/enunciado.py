"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  -  FCL1109                            ║
║                    PARCIAL III  ·  SIMULACRO  F                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROBLEMA: Propagación de Ondas Electromagnéticas en 1D - Método FDTD
  ────────────────────────────────────────────────────────────────────

  Una onda electromagnética que viaja en la dirección z (con el campo eléctrico
  en x y el magnético en y) se rige, en 1D y unidades con c = 1, por las dos
  ecuaciones de Maxwell del rotacional:

        ∂E_x/∂t = −c ∂B_y/∂z
        ∂B_y/∂t = −c ∂E_x/∂z

  Combinándolas se obtiene la ecuación de onda  ∂²E_x/∂t² = c² ∂²E_x/∂z²,
  de modo que cualquier perturbación viaja a la velocidad de la luz c.

  MÉTODO FDTD (Finite-Difference Time-Domain, esquema de Yee / "leapfrog"):
  ─────────────────────────────────────────────────────────────────────────
  Discretizamos z en pasos Δz y el tiempo en pasos Δt.  Aproximando las
  derivadas por diferencias finitas y definiendo el número de Courant
  β = c·Δt/Δz, las ecuaciones se vuelven una receta de actualización que
  alterna E y B (uno se calcula con el otro ya actualizado → "salto de rana"):

        E_x[i] ← E_x[i] + β (B_y[i−1] − B_y[i])
        B_y[i] ← B_y[i] − β (E_x[i+1] − E_x[i])

  En un MEDIO DIELÉCTRICO de permitividad relativa ε_r, el campo eléctrico
  evoluciona más lento (la luz va a v = c/√ε_r):

        E_x[i] ← E_x[i] + (β/ε_r[i]) (B_y[i−1] − B_y[i])

  CONDICIÓN DE ESTABILIDAD (Courant-Friedrichs-Lewy, 1D):
        β = c·Δt/Δz ≤ 1
  Si β > 1 la solución numérica DIVERGE (crece sin control), aunque la física
  sea estable: es el mismo fenómeno que viste en la ecuación de onda 1D.

  Condiciones de frontera (paredes conductoras perfectas, PEC):
        E_x = 0  y  B_y = 0  en los extremos z = 0 y z = final.

  NOTA: NO se piden animaciones.  Como en el parcial 2, se capturan
  "fotografías" (snapshots) del campo en instantes fijos y se grafican como
  curvas estáticas superpuestas para visualizar la evolución.

  Parámetros base:  c = 1,  Nz = 300,  Δz = 1,  β = 0.4,  pulso gaussiano
  centrado en z₀ = 80 con ancho σ = 15.

  ─────────────────────────────────────────────────────────────────────────────
  TAREAS
  ─────────────────────────────────────────────────────────────────────────────

  1. Implemente la actualización FDTD de E_x y B_y (esquema leapfrog) con
     condiciones de frontera de pared conductora (E = B = 0 en los extremos).
     Explique por qué E y B se actualizan de forma ALTERNADA y qué representa
     físicamente el número de Courant β.

  2. Inicialice un pulso gaussiano en E_x con B_y = 0 y propáguelo.  Capture
     snapshots en varios instantes y grafíquelos.  ¿Qué ocurre con el pulso?
     (Sugerencia: un estado con E ≠ 0 y B = 0 es la superposición de dos ondas
     viajeras; observe en cuántos pulsos se divide y con qué amplitud.)

  3. Estudie la CONDICIÓN DE COURANT.  Repita la simulación con β = 0.4
     (estable) y β = 1.05 (inestable).  Grafique el máximo de |E_x| en función
     del tiempo (escala semilog).  ¿Qué β garantiza estabilidad y por qué?

  4. Reflexión en una pared conductora: lance un pulso viajero (B_y = E_x, que
     se mueve en +z) contra la pared en z = final.  Capture snapshots antes y
     después del rebote.  ¿Cambia de signo el campo al reflejarse?  ¿Por qué lo
     exige la condición E = 0 en un conductor perfecto?

  5. Lámina dieléctrica: coloque un medio con ε_r = 4 en una región central y
     lance un pulso viajero hacia ella.  Capture snapshots.  Identifique la
     parte REFLEJADA y la TRANSMITIDA, y verifique que dentro del dieléctrico la
     onda viaja más lento (v = c/√ε_r) y con menor longitud de onda.

  6. Energía y vector de Poynting.  Calcule la densidad de energía
     u(z) = ½(ε_r E_x² + B_y²) y el flujo de Poynting S_z = E_x·B_y.
     Verifique que la energía total se conserva (con paredes PEC) y que el signo
     de S_z indica la dirección de propagación.
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parámetros físicos y numéricos ──────────────────────────────────────────
c      = 1.0      # velocidad de la luz (unidades reducidas)
Nz     = 300      # número de nodos espaciales
dz     = 1.0      # paso espacial
beta   = 0.4      # número de Courant β = c·Δt/Δz  (debe ser ≤ 1)
dt     = beta * dz / c
pasos  = 600      # número de pasos de tiempo

# Pulso gaussiano inicial
z0     = 80.0     # centro del pulso
sigma  = 15.0     # ancho del pulso

z = np.arange(Nz)

# ─── ÍTEM 1: Actualización FDTD ──────────────────────────────────────────────

def paso_fdtd(E_x, B_y, beta, eps_r):
    # TODO: actualizar E_x[1:-1] += (beta/eps_r[1:-1])*(B_y[:-2] - B_y[1:-1])
    # TODO: actualizar B_y[1:-1] += -beta*(E_x[2:] - E_x[1:-1])
    # TODO: imponer E_x[0]=E_x[-1]=0 y B_y[0]=B_y[-1]=0
    pass

# ─── ÍTEM 2: Pulso gaussiano (B=0) y snapshots ───────────────────────────────
# TODO: inicializar E_x gaussiano, B_y=0; iterar guardando snapshots; graficar

# ─── ÍTEM 3: Condición de Courant ────────────────────────────────────────────
# TODO: correr con beta=0.4 y beta=1.05; graficar max|E| vs tiempo (semilogy)

# ─── ÍTEM 4: Reflexión en pared conductora ───────────────────────────────────
# TODO: pulso viajero (B_y=E_x) hacia la pared; snapshots del rebote

# ─── ÍTEM 5: Lámina dieléctrica ──────────────────────────────────────────────
# TODO: eps_r=4 en una región; pulso viajero; snapshots de reflexión/transmisión

# ─── ÍTEM 6: Energía y vector de Poynting ────────────────────────────────────
# TODO: u = 0.5*(eps_r*E**2 + B**2); S = E*B; energía total vs tiempo

plt.show()
