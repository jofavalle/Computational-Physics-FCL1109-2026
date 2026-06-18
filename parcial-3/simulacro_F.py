"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR — FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  —  FCO4101                            ║
║                    PARCIAL III  ·  SIMULACRO  F  —  SOLUCIÓN                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Propagación de ondas electromagnéticas en 1D con el método FDTD
  (Finite-Difference Time-Domain), esquema "leapfrog" de Yee.

  Física (1D, c = 1):   ∂E_x/∂t = −c ∂B_y/∂z ,   ∂B_y/∂t = −c ∂E_x/∂z
  ⇒ ecuación de onda    ∂²E_x/∂t² = c² ∂²E_x/∂z²  (todo viaja a velocidad c).

  Receta numérica (β = c·Δt/Δz = número de Courant):
        E_x[i] ← E_x[i] + (β/ε_r[i]) (B_y[i−1] − B_y[i])
        B_y[i] ← B_y[i] − β (E_x[i+1] − E_x[i])
  Paredes conductoras (PEC):  E_x = B_y = 0 en los extremos.
  Estabilidad (Courant 1D):   β ≤ 1.

  IDEA DEL CÓDIGO: en vez de animar, integramos en el tiempo y vamos guardando
  "fotografías" (snapshots) del campo en instantes elegidos; al final las
  dibujamos como curvas estáticas superpuestas (igual que en el parcial 2).
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Importar librerías ────────────────────────────────────────────────────────
import numpy as np               # arreglos y matemáticas (np.exp, np.zeros, ...)
import matplotlib.pyplot as plt  # gráficas

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros físicos y numéricos
# ══════════════════════════════════════════════════════════════════════════════
c     = 1.0      # velocidad de la luz (unidades reducidas)
Nz    = 300      # número de nodos espaciales (la "rejilla" del eje z)
dz    = 1.0      # paso espacial (distancia entre nodos)
beta  = 0.4      # número de Courant β = c·Δt/Δz. Debe ser ≤ 1 para estabilidad.
dt    = beta * dz / c    # paso de tiempo despejado de β

z     = np.arange(Nz)    # np.arange(Nz) = [0, 1, 2, ..., Nz-1]: posiciones de los nodos

# Pulso gaussiano inicial: una "joroba" suave centrada en z0, de ancho sigma.
z0    = 80.0     # centro del pulso
sigma = 15.0     # ancho del pulso

def pulso_gaussiano():
    """Devuelve un pulso gaussiano exp(-((z-z0)/sigma)²) sobre toda la rejilla.
    np.exp aplica la exponencial a cada elemento del arreglo de una sola vez."""
    return np.exp(-((z - z0) / sigma) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 1 — Paso de actualización FDTD (esquema leapfrog)
# ══════════════════════════════════════════════════════════════════════════════
# ¿Por qué se actualizan E y B de forma ALTERNADA?
# ------------------------------------------------
# Las ecuaciones de Maxwell acoplan E y B: el cambio temporal de E depende de la
# variación espacial de B y viceversa. El esquema de Yee aprovecha esto: primero
# avanza E usando el B actual, e inmediatamente avanza B usando el E recién
# actualizado. Es un "salto de rana" (leapfrog): cada campo salta sobre el otro,
# medio paso de tiempo desfasado. Esto es estable y de 2º orden de precisión.
#
# El número de Courant β = c·Δt/Δz mide CUÁNTOS nodos avanza la luz en un paso de
# tiempo. Si β > 1 la onda querría saltarse más de un nodo por paso y el esquema
# (que sólo "ve" a sus vecinos inmediatos) no puede seguirla → diverge.

def paso_fdtd(E_x, B_y, beta, eps_r):
    """Avanza los campos UN paso de tiempo, MODIFICÁNDOLOS en el sitio (in-place).
    Las "rebanadas" de NumPy aplican la fórmula a todos los nodos interiores a la
    vez (sin bucles 'for' lentos):
      B_y[:-2]  = vecino izquierdo (i−1),  B_y[1:-1] = nodo (i)
      E_x[2:]   = vecino derecho  (i+1),  E_x[1:-1] = nodo (i)."""
    # 1) Avanzar E_x con el B_y actual. Dividir por ε_r hace que dentro de un
    #    dieléctrico el campo evolucione más lento (velocidad v = c/√ε_r).
    E_x[1:-1] += (beta / eps_r[1:-1]) * (B_y[:-2] - B_y[1:-1])
    # 2) Avanzar B_y con el E_x ya actualizado (de ahí el "leapfrog").
    B_y[1:-1] += -beta * (E_x[2:] - E_x[1:-1])
    # 3) Condiciones de frontera de pared conductora perfecta (PEC):
    #    el campo no puede existir dentro de un conductor ideal → 0 en los bordes.
    E_x[0] = E_x[-1] = 0.0
    B_y[0] = B_y[-1] = 0.0


def simular(E_x, B_y, beta, eps_r, pasos, instantes):
    """Integra 'pasos' veces y guarda fotografías de (E_x, B_y) en los números de
    paso indicados en 'instantes'. Devuelve un diccionario {paso: (E, B)} y la
    lista del máximo de |E_x| en cada paso (útil para ver estabilidad)."""
    snaps = {}                       # diccionario vacío para las fotos
    max_E = []                       # lista del pico de |E| en el tiempo
    if 0 in instantes:               # foto del estado inicial, si se pide
        snaps[0] = (E_x.copy(), B_y.copy())   # .copy() para guardar una foto independiente
    for n in range(1, pasos + 1):
        paso_fdtd(E_x, B_y, beta, eps_r)      # un paso de tiempo
        max_E.append(np.max(np.abs(E_x)))     # registrar el pico de |E_x|
        if n in instantes:
            snaps[n] = (E_x.copy(), B_y.copy())
    return snaps, max_E


# Medio "vacío" por defecto: ε_r = 1 en todos los nodos.
# np.ones(Nz) crea un arreglo de Nz unos.
eps_vacio = np.ones(Nz)


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 2 — Pulso gaussiano con B = 0: se divide en dos
# ══════════════════════════════════════════════════════════════════════════════
# Estado inicial: E_x gaussiano, B_y = 0. Físicamente, un campo con E≠0 y B=0 NO
# es una onda viajera pura, sino la SUMA de dos ondas iguales que viajan en +z y
# en −z. Por eso el pulso se parte en dos jorobas de la MITAD de amplitud que se
# alejan en sentidos opuestos.
print("═" * 64)
print("ÍTEM 2 — Pulso gaussiano con B=0 (se divide en dos)")
print("═" * 64)

E_x = pulso_gaussiano()              # E inicial = gaussiana
B_y = np.zeros(Nz)                   # B inicial = 0
# Tomamos las fotos antes de que ninguna mitad llegue a las paredes (la reflexión
# es el tema del ítem 4); así se ve LIMPIO cómo el pulso se parte en dos.
instantes2 = [0, 40, 80, 120]        # números de paso donde tomar fotos
snaps2, _ = simular(E_x, B_y, beta, eps_vacio, pasos=120, instantes=instantes2)
print(f"  Amplitud inicial = {pulso_gaussiano().max():.2f}")
# Tras dividirse, cada mitad tiene la MITAD de la amplitud (resultado de d'Alembert):
print(f"  Amplitud de cada mitad (paso 120) ≈ {snaps2[120][0].max():.2f}")

# Gráfica: una curva por snapshot, todas superpuestas.
plt.figure(figsize=(11, 5))          # crea una figura nueva
for n in instantes2:
    E_snap = snaps2[n][0]            # [0] toma E_x del par (E_x, B_y)
    plt.plot(z, E_snap, label=f'paso {n}  (t = {n*dt:.1f})')   # dibuja E(z) en ese instante
plt.axhline(0, color='k', lw=0.6)    # eje horizontal en 0
plt.title('ÍTEM 2 — Pulso con B=0: se separa en dos ondas (±z) de media amplitud')
plt.xlabel('z'); plt.ylabel(r'$E_x$')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 3 — Condición de Courant: estable (β=0.4) vs inestable (β=1.05)
# ══════════════════════════════════════════════════════════════════════════════
# Repetimos la simulación con dos β. Con β≤1 el pico de |E| se mantiene acotado;
# con β>1 crece exponencialmente (el esquema "explota"): pura inestabilidad
# numérica, no física.
print("\n" + "═" * 64)
print("ÍTEM 3 — Condición de Courant (estabilidad)")
print("═" * 64)

plt.figure(figsize=(11, 5))
for b, color in [(0.4, 'tab:blue'), (1.05, 'tab:red')]:
    E_x = pulso_gaussiano()
    B_y = np.zeros(Nz)
    _, max_E = simular(E_x, B_y, b, eps_vacio, pasos=400, instantes=[])
    estado = "ESTABLE" if b <= 1 else "INESTABLE (diverge)"
    print(f"  β = {b}: max|E| final = {max_E[-1]:.3e}  → {estado}")
    # semilogy: eje Y logarítmico. Ideal para ver crecer/decaer en órdenes de magnitud.
    plt.semilogy(max_E, color=color, lw=2, label=f'β = {b}  ({estado})')
plt.title('ÍTEM 3 — Estabilidad de Courant:  β ≤ 1 acotado,  β > 1 diverge')
plt.xlabel('paso de tiempo'); plt.ylabel(r'$\max|E_x|$  (escala log)')
plt.legend(); plt.grid(alpha=0.3, which='both')
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 4 — Reflexión en una pared conductora (PEC)
# ══════════════════════════════════════════════════════════════════════════════
# Ahora lanzamos una onda VIAJERA pura hacia la derecha. El truco para que viaje
# sólo en +z es arrancar con B_y = E_x (los dos campos en fase): así una de las
# dos componentes (±z) tiene amplitud cero. Al chocar con la pared z=final, donde
# se obliga E=0, el pulso se refleja INVIRTIENDO su signo (como una cuerda atada
# a un extremo fijo).
print("\n" + "═" * 64)
print("ÍTEM 4 — Reflexión en pared conductora (inversión de signo)")
print("═" * 64)

E_x = pulso_gaussiano()
B_y = E_x.copy()                     # B = E  →  onda que viaja en +z
# El pulso parte en z0=80 y la pared está en z≈299: tarda ~(299-80)/β ≈ 547 pasos
# en llegar. Por eso simulamos 800 pasos para ver el viaje, el choque y el regreso.
instantes4 = [0, 350, 550, 750]
snaps4, _ = simular(E_x, B_y, beta, eps_vacio, pasos=800, instantes=instantes4)
print(f"  E_x en el pico inicial   (paso 0)   = {snaps4[0][0].max():+.2f}")
print(f"  E_x extremo tras rebotar (paso 750) = {snaps4[750][0].min():+.2f}  (signo invertido)")

plt.figure(figsize=(11, 5))
for n in instantes4:
    plt.plot(z, snaps4[n][0], label=f'paso {n}')
plt.axhline(0, color='k', lw=0.6)
plt.axvline(Nz - 1, color='gray', ls='--', label='pared conductora')
plt.title('ÍTEM 4 — El pulso rebota en la pared y se INVIERTE (E=0 en el conductor)')
plt.xlabel('z'); plt.ylabel(r'$E_x$')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 5 — Lámina dieléctrica: reflexión y transmisión
# ══════════════════════════════════════════════════════════════════════════════
# Colocamos un medio con ε_r = 4 en una franja central. Cuando la onda viajera
# llega a la interfaz:
#   · parte se REFLEJA (vuelve en −z),
#   · parte se TRANSMITE (sigue en +z) pero más LENTA: v = c/√ε_r = c/2,
#     y con menor longitud de onda (se "comprime" dentro del dieléctrico).
print("\n" + "═" * 64)
print("ÍTEM 5 — Lámina dieléctrica (reflexión + transmisión)")
print("═" * 64)

eps_r = np.ones(Nz)                  # arrancamos en vacío (ε_r = 1) en todos lados
ini_diel, fin_diel = 150, 210        # la lámina ocupa los nodos [150, 210)
eps_r[ini_diel:fin_diel] = 4.0       # ε_r = 4 dentro de la lámina
n_diel = np.sqrt(4.0)                 # índice de refracción n = √ε_r = 2
# Coeficiente de reflexión de Fresnel (incidencia normal): r = (1−n)/(1+n).
r_fresnel = (1 - n_diel) / (1 + n_diel)
print(f"  Dieléctrico ε_r = 4 en z ∈ [{ini_diel}, {fin_diel})")
print(f"  Velocidad dentro = c/√ε_r = c/{n_diel:.0f} = {c/n_diel:.2f}")
print(f"  Coef. de reflexión teórico r = (1−n)/(1+n) = {r_fresnel:+.3f}  (n = {n_diel:.0f})")

E_x = pulso_gaussiano()
B_y = E_x.copy()                     # onda viajera hacia +z
# La onda tarda ~300 pasos en CRUZAR la lámina (dentro va a c/2). Simulamos 520
# para ver entrar, reflejarse y salir transmitida por el otro lado.
instantes5 = [0, 150, 300, 520]
snaps5, _ = simular(E_x, B_y, beta, eps_r, pasos=520, instantes=instantes5)
print(f"  Amplitud reflejada medida ≈ {snaps5[300][0][:ini_diel].min():+.2f}"
      f"  (coincide con r = {r_fresnel:+.2f})")

plt.figure(figsize=(11, 5))
# axvspan pinta una banda vertical que marca la región del dieléctrico.
plt.axvspan(ini_diel, fin_diel, color='gold', alpha=0.25, label='dieléctrico ε_r=4')
for n in instantes5:
    plt.plot(z, snaps5[n][0], label=f'paso {n}')
plt.axhline(0, color='k', lw=0.6)
plt.title('ÍTEM 5 — Onda contra dieléctrico: parte se refleja, parte se transmite (más lenta)')
plt.xlabel('z'); plt.ylabel(r'$E_x$')
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 6 — Energía y vector de Poynting
# ══════════════════════════════════════════════════════════════════════════════
# Densidad de energía:  u(z) = ½(ε_r E² + B²)   (eléctrica + magnética).
# Flujo de Poynting:     S_z = E_x · B_y  → su SIGNO dice hacia dónde fluye la
# energía (+z si E y B están en fase, −z si en contrafase).
# Con paredes PEC ideales la energía total ∫u dz se conserva (la onda rebota sin
# perderse). La medimos a lo largo del tiempo para comprobarlo.
print("\n" + "═" * 64)
print("ÍTEM 6 — Energía total y vector de Poynting")
print("═" * 64)

E_x = pulso_gaussiano()
B_y = E_x.copy()                     # onda viajera (+z): debe dar Poynting > 0
energias = []                        # energía total en cada paso
for n in range(300):
    paso_fdtd(E_x, B_y, beta, eps_vacio)
    u = 0.5 * (eps_vacio * E_x**2 + B_y**2)   # densidad de energía en cada nodo
    energias.append(np.sum(u) * dz)           # energía total = ∫u dz ≈ Σ u·Δz
S_z = E_x * B_y                      # vector de Poynting (a lo largo de z)
print(f"  Energía total inicial = {energias[0]:.3f}")
print(f"  Energía total final   = {energias[-1]:.3f}   (≈ constante ⇒ se conserva)")
print(f"  Poynting medio  <S_z> = {np.mean(S_z):+.3f}  → signo + = propagación en +z")

# Dos paneles: energía total vs tiempo, y Poynting S_z(z) en el último instante.
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(energias, 'tab:green', lw=2)
ax[0].set(xlabel='paso de tiempo', ylabel='energía total',
          title='ÍTEM 6 — Energía total (se conserva con paredes PEC)')
# ylim ajustado para apreciar que la curva es esencialmente plana:
ax[0].set_ylim(0, max(energias) * 1.3)
ax[0].grid(alpha=0.3)

ax[1].plot(z, S_z, 'tab:purple', lw=2)
ax[1].fill_between(z, S_z, color='tab:purple', alpha=0.25)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set(xlabel='z', ylabel=r'$S_z = E_x B_y$',
          title='Flujo de Poynting (signo + ⇒ energía hacia +z)')
ax[1].grid(alpha=0.3)
plt.tight_layout()

plt.show()   # abre todas las figuras (al final del programa)
