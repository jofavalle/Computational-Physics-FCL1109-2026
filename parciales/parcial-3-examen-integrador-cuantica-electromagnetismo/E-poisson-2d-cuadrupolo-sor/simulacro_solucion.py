"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSIDAD DE EL SALVADOR - FACULTAD DE CIENCIAS NATURALES            ║
║                    FÍSICA COMPUTACIONAL  -  FCL1109                            ║
║                    PARCIAL III  ·  SIMULACRO  E  -  SOLUCIÓN                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Cuadrupolo de placas cargadas + ELECTRODO CENTRAL a potencial fijo (500 V),
  dentro de una caja conductora aterrizada (V = 0 en las paredes).

  Resolvemos la ecuación de Poisson  ∇²V = −ρ/ε₀  con condiciones de Dirichlet
  MIXTAS (paredes a 0 V, electrodo a 500 V) mediante diferencias finitas y
  sobrerelajación sucesiva (SOR).  Esta combinación -carga prescrita (ρ) +
  potencial prescrito en conductores- es la base del cálculo de capacitores.

  IDEA GENERAL: discretizamos el espacio en una rejilla N×N de puntos (nodos).
  El potencial V deja de ser una función continua y pasa a ser una MATRIZ V[i,j]
  (i = fila, j = columna). La ecuación de Poisson se convierte en una relación
  entre cada nodo y sus 4 vecinos, que resolvemos iterando hasta que V deja de
  cambiar (converge).

  Discretización (paso Δ):
      V[i,j] = ¼(V[i+1,j]+V[i-1,j]+V[i,j+1]+V[i,j-1]) + Δ²ρ[i,j]/(4ε₀)     (*)
  Es decir: en equilibrio, cada nodo es el PROMEDIO de sus 4 vecinos (más el
  término de carga). Esto sale de aproximar las segundas derivadas de ∇²V.

  Actualización SOR:
      V[i,j] ← (1−ω)·V[i,j] + ω·V_GS[i,j]                                  (**)
  Los nodos con potencial prescrito (paredes + electrodo) NO se actualizan.
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Importar librerías ────────────────────────────────────────────────────────
import numpy as np               # arreglos/matrices y matemáticas
import matplotlib.pyplot as plt  # gráficas (incluye superficies 3D)

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros del problema
# ══════════════════════════════════════════════════════════════════════════════
N        = 100          # nodos por lado → la rejilla es de 100×100 puntos
delta    = 1.0          # [u.r.] paso de malla (distancia entre nodos vecinos)
epsilon0 = 1.0          # permitividad ε₀ (unidades reducidas)
rho0     = 10.0         # [u.r.] densidad de carga de las placas
tol      = 1e-5         # criterio de convergencia: paramos cuando V cambia menos que esto
N_iter   = 20000        # tope de iteraciones (red de seguridad por si no converge)

omega    = 1.85         # factor de sobrerelajación ω por defecto (acelera la convergencia)
omega_opt = 2.0 / (1.0 + np.pi / N)   # ω óptimo teórico para esta malla ≈ 1.939

# Electrodo central a potencial fijo (la novedad: una superficie a V constante)
cx = cy = N // 2        # centro de la caja. '//' = división entera → nodo (50, 50)
R_elec  = 8             # [nodos] radio del electrodo circular
V_elec  = 500.0         # [V] potencial fijo del electrodo

# Geometría de las cuatro placas. Cada entrada es una tupla:
#   (rango_de_filas, rango_de_columnas, carga, color_para_dibujar, etiqueta)
# 'slice(15, 35)' representa el rango de índices 15,16,...,34 (como V[15:35]).
PLACAS = [(slice(15, 35), slice(15, 35), +rho0, 'red',  '+ρ₀'),   # sup-izq
          (slice(15, 35), slice(65, 85), -rho0, 'blue', '−ρ₀'),   # sup-der
          (slice(65, 85), slice(15, 35), -rho0, 'blue', '−ρ₀'),   # inf-izq
          (slice(65, 85), slice(65, 85), +rho0, 'red',  '+ρ₀')]   # inf-der


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 1 - Densidad de carga, electrodo y condiciones de contorno
# ══════════════════════════════════════════════════════════════════════════════
# Sistema físico: el cuadrupolo (dos cargas + y dos −) es una fuente de carga
# (término ρ de Poisson); el electrodo central es un CONDUCTOR a potencial fijo
# (condición de Dirichlet interior).  Junto con la caja aterrizada, electrodo y
# pared forman las dos armaduras de un CAPACITOR: una a 500 V, la otra a 0 V.
#
# Superposición (Poisson es lineal):  V = V_cuadrupolo + V_capacitor, donde
#   · V_cuadrupolo: fuente ρ, con paredes Y electrodo a 0 V  → ANTISIMÉTRICO
#     (cambia de signo bajo rotación de 90°; se anula en las diagonales).
#   · V_capacitor:  sin carga (ρ=0), paredes a 0 y electrodo a 500 V → SIMÉTRICO
#     (positivo en todo el interior; un "pedestal" decreciente hacia la pared).
# La suma ya NO es antisimétrica: el electrodo rompe la simetría del cuadrupolo.

def construir_fuentes():
    """Prepara los datos del problema. Devuelve cuatro matrices N×N:
       rho       : densidad de carga en cada nodo
       electrodo : máscara True/False de los nodos que son el electrodo
       fija      : máscara de TODOS los nodos con potencial fijo (paredes+electrodo)
       V_dir     : valores fijos del potencial (0 en paredes, 500 en el electrodo)."""
    rho = np.zeros((N, N))               # matriz N×N de ceros (sin carga al inicio)
    for fila, col, q, _, _ in PLACAS:    # recorremos las 4 placas (desempaquetando la tupla)
        rho[fila, col] = q               # rho[15:35, 15:35] = +rho0, etc. (asigna a un bloque)

    # Máscara circular del electrodo. np.meshgrid construye dos matrices ii, jj con
    # la fila y la columna de cada nodo (las "coordenadas" de cada celda). Así
    # podemos evaluar la condición del círculo en TODA la rejilla de una sola vez.
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    electrodo = (ii - cx)**2 + (jj - cy)**2 <= R_elec**2   # True dentro del círculo (distancia² ≤ R²)

    # Nodos con potencial prescrito (Dirichlet): el electrodo + las 4 paredes.
    fija = electrodo.copy()              # .copy() para no modificar 'electrodo' al tocar 'fija'
    # V[0,:] = primera fila; V[-1,:] = última fila; V[:,0] y V[:,-1] = columnas borde.
    fija[0, :] = fija[-1, :] = fija[:, 0] = fija[:, -1] = True

    # Valores de Dirichlet: 0 en las paredes (ya están en 0) y V_elec en el electrodo.
    V_dir = np.zeros((N, N))
    V_dir[electrodo] = V_elec            # asigna 500 sólo a los nodos donde electrodo==True
    return rho, electrodo, fija, V_dir


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 2 - Actualización SOR (esquema rojo-negro vectorizado)
# ══════════════════════════════════════════════════════════════════════════════
# Jacobi / Gauss-Seidel / SOR (tres formas de iterar el promedio de vecinos):
#   · Jacobi: cada V[i,j] nuevo usa SOLO valores de la iteración anterior →
#     la información se propaga un nodo por iteración (convergencia O(N²) lenta).
#   · Gauss-Seidel (GS): usa los valores ya actualizados en el mismo barrido →
#     ~2× más rápido que Jacobi, pero aún O(N²).
#   · SOR: extrapola más allá de GS, V ← (1−ω)V + ω·V_GS.  Con el ω óptimo la
#     convergencia pasa a O(N): drásticamente más rápido.  Si ω ≥ 2 el radio
#     espectral del iterador supera 1 y el método DIVERGE (sobre-sobre-relajación).
#
# Implementamos SOR con ordenamiento ROJO-NEGRO (tablero de ajedrez): se
# actualizan primero los nodos (i+j) par usando a sus vecinos (todos impares),
# y luego los impares usando los pares ya renovados.  Esto reproduce el carácter
# "Gauss-Seidel" (usa info fresca) PERO permite VECTORIZAR con NumPy: en vez de
# dos bucles 'for' recorriendo nodo por nodo (lentísimo en Python), operamos
# sobre matrices completas de un golpe (NumPy lo hace en C, muy rápido).
# Es el mismo método nodo-a-nodo del enunciado, sólo que mucho más eficiente.
#
# Los nodos prescritos (paredes + electrodo) NUNCA se tocan: fijar el electrodo
# a V_elec es EXACTAMENTE la condición de un conductor en equilibrio (potencial
# constante en su volumen, de modo que E = −∇V = 0 en su interior).

def resolver_sor(omega, rho, fija, V_dir, tol=tol, N_iter=N_iter):
    """Resuelve Poisson con SOR rojo-negro. Devuelve (V, historial_error)."""
    V = V_dir.copy()                       # arranca con las CC ya impuestas (no toca V_dir original)
    fuente = delta**2 * rho / epsilon0     # término Δ²ρ/ε₀ de (*), precalculado una vez

    # Construimos las máscaras de qué nodos actualizar y de qué color son.
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    interior = np.zeros((N, N), bool)      # matriz de False
    interior[1:-1, 1:-1] = True            # True en todo menos el borde (1:-1 = "del 2º al penúltimo")
    libre    = interior & ~fija            # '&' = Y lógico; '~' = NO. Nodos interiores y NO fijos.
    rojo     = libre & (((ii + jj) % 2) == 0)   # '% 2 == 0' → suma de índices par (casillas rojas)
    negro    = libre & (((ii + jj) % 2) == 1)   # suma impar (casillas negras)

    historial = []                         # aquí guardamos el error de cada iteración
    for it in range(N_iter):
        V_old = V.copy()                   # copia para medir cuánto cambió V en esta iteración
        for color in (rojo, negro):        # primero los rojos, luego los negros
            # Suma de los 4 vecinos de CADA nodo interior, calculada con "rebanadas":
            #   V[2:,1:-1]   = vecino de ABAJO (fila i+1)
            #   V[:-2,1:-1]  = vecino de ARRIBA (fila i-1)
            #   V[1:-1,2:]   = vecino DERECHO  (columna j+1)
            #   V[1:-1,:-2]  = vecino IZQUIERDO(columna j-1)
            # Sumarlas desplazadas equivale a hacer V[i±1,j±1] para todos los i,j a la vez.
            vecinos = np.zeros((N, N))
            vecinos[1:-1, 1:-1] = (V[2:, 1:-1] + V[:-2, 1:-1] +
                                   V[1:-1, 2:] + V[1:-1, :-2])
            V_GS = 0.25 * (vecinos + fuente)        # valor de Gauss-Seidel, ec. (*)
            # Actualización SOR (**) aplicada SÓLO a los nodos de este color.
            # V[color] selecciona esos nodos; les asignamos su valor nuevo de golpe.
            V[color] = (1.0 - omega) * V[color] + omega * V_GS[color]

        # Error = el mayor cambio de cualquier nodo respecto a la iteración previa.
        # np.abs = valor absoluto (elemento a elemento); np.max = el máximo de toda la matriz.
        error = np.max(np.abs(V - V_old))
        historial.append(error)
        if error < tol:                    # convergió: V ya casi no cambia
            break                          # salimos del bucle
    return V, historial


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 3 - Resolver hasta convergencia y registrar el historial
# ══════════════════════════════════════════════════════════════════════════════
# Llamamos a las funciones. La asignación múltiple desempaqueta las 4 matrices.
rho, electrodo, fija, V_dir = construir_fuentes()

print("═" * 64)
print(f"ÍTEM 3 - Convergencia SOR  (ω = {omega},  tol = {tol})")
print("═" * 64)
V, hist = resolver_sor(omega, rho, fija, V_dir)    # resolvemos con el ω por defecto
print(f"  Convergencia en {len(hist)} iteraciones.")     # len(hist) = nº de iteraciones hechas
print(f"  V máximo = {V.max():.1f} V   ·   V mínimo = {V.min():.1f} V")   # .max()/.min() de la matriz
print(f"  V en el centro (electrodo) = {V[cx, cy]:.1f} V  (debe ser {V_elec:.0f})")
print(f"  ω óptimo teórico 2/(1+π/N) = {omega_opt:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 5 (cálculo) - Campo eléctrico  E = −∇V
# ══════════════════════════════════════════════════════════════════════════════
# El campo es menos el gradiente del potencial. Aproximamos las derivadas por
# "diferencias centrales": ∂V/∂x ≈ (V[j+1] − V[j-1]) / (2Δ) (pendiente usando los
# vecinos a ambos lados). Lo hacemos para toda la matriz a la vez con rebanadas.
# Convención: x = dirección de columnas (j), y = dirección de filas (i).
Ex = np.zeros_like(V)    # np.zeros_like(V) = matriz de ceros del mismo tamaño que V
Ey = np.zeros_like(V)
Ex[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2 * delta)   # −∂V/∂x (deriva en columnas)
Ey[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * delta)   # −∂V/∂y (deriva en filas)
E_mag = np.hypot(Ex, Ey)   # np.hypot(a,b) = √(a²+b²): magnitud del campo en cada nodo


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 6 - Carga y capacitancia del electrodo (ley de Gauss en 2D)
# ══════════════════════════════════════════════════════════════════════════════
# Por Gauss:  Q_enc/ε₀ = ∮ E·n̂ dl  sobre un lazo cerrado.  Tomamos un cuadrado
# que rodea SÓLO al electrodo (sin encerrar las placas, que están en índices
# ≤35 y ≥65).  El lazo r0..r1 = 36..64 es región sin carga libre, así que el
# flujo total que sale del lazo mide la carga encerrada (la del electrodo).
r0, r1 = 36, 64
filas = np.arange(r0, r1 + 1)   # np.arange(a, b) = [a, a+1, ..., b-1]; +1 para incluir r1
cols  = np.arange(r0, r1 + 1)
# Sumamos la componente del campo PERPENDICULAR a cada lado del cuadrado (su flujo).
# np.sum suma todos los elementos del arreglo. Cada lado contribuye con su normal:
#   lado derecho (columna r1): normal +x → +Ex ;  lado izquierdo (col r0): normal −x → −Ex
#   lado superior (fila r1):  normal +y → +Ey ;  lado inferior (fila r0): normal −y → −Ey
flujo = delta * (
    np.sum(Ex[filas, r1]) - np.sum(Ex[filas, r0]) +
    np.sum(Ey[r1, cols])  - np.sum(Ey[r0, cols])
)
Q_elec = epsilon0 * flujo       # carga encerrada = ε₀ × flujo
C_elec = Q_elec / V_elec        # capacitancia C = Q / V (definición de capacitor)

print("\n" + "═" * 64)
print("ÍTEM 6 - Carga y capacitancia del electrodo (ley de Gauss 2D)")
print("═" * 64)
print(f"  Flujo ∮E·n̂ dl alrededor del electrodo = {flujo:.3f}")
print(f"  Carga del electrodo  Q_e = ε₀·flujo    = {Q_elec:.3f} (u.r.)")
print(f"  Capacitancia         C   = Q_e / V_e   = {C_elec:.5f} (u.r.)")
print("  Q_e > 0: el electrodo positivo emite líneas de campo hacia la caja")
print("  aterrizada (su carga inducida negativa) → es un capacitor de placas")
print("  concéntricas (electrodo-caja) con el cuadrupolo como perturbación.")


# ══════════════════════════════════════════════════════════════════════════════
# ÍTEM 7 - Comparación de eficiencia entre valores de ω
# ══════════════════════════════════════════════════════════════════════════════
# Resolvemos el MISMO problema con tres ω distintos y comparamos cuántas
# iteraciones tarda cada uno. Como 'resolver_sor' es una función reutilizable,
# basta con llamarla en un bucle.
print("\n" + "═" * 64)
print("ÍTEM 7 - Eficiencia vs. ω")
print("═" * 64)
historiales = {}    # diccionario: asociará cada ω con su lista de errores {ω: historial}
for w in (1.0, 1.85, 1.95):
    _, h = resolver_sor(w, rho, fija, V_dir)   # '_' descarta la matriz V (sólo queremos el historial)
    historiales[w] = h                          # guardamos el historial bajo la clave w
    etiqueta = "Gauss-Seidel" if w == 1.0 else "SOR"   # ω=1.0 es justo Gauss-Seidel
    print(f"  ω = {w:.2f}  ({etiqueta:<12}) → {len(h):5d} iteraciones")
print(f"  ω óptimo teórico ≈ {omega_opt:.3f}: minimiza el nº de iteraciones.")
# min(diccionario, key=...) busca la clave ω cuyo historial sea el más corto.
mejor = min(historiales, key=lambda w: len(historiales[w]))
print(f"  Más eficiente de los probados: ω = {mejor}  "
      f"({len(historiales[mejor])} iter, frente a {len(historiales[1.0])} de GS).")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS  (Ítems 3, 4, 5, 7)
# ══════════════════════════════════════════════════════════════════════════════
# Coordenadas físicas de los nodos para los ejes de las gráficas.
x = np.arange(N) * delta        # [0, Δ, 2Δ, ...] posiciones en x
y = np.arange(N) * delta
# np.meshgrid arma las matrices X, Y con la coordenada (x,y) de cada nodo, que es
# lo que necesitan contourf/plot_surface para ubicar cada valor V[i,j] en el plano.
X, Y = np.meshgrid(x, y)

def dibujar_placas(ax):
    """Dibuja el contorno de las 4 placas y el círculo del electrodo sobre el eje 'ax'."""
    for fila, col, _, color, _ in PLACAS:
        # Las 5 esquinas (la 1ª se repite al final para cerrar el rectángulo).
        xs = [col.start, col.stop, col.stop, col.start, col.start]
        ys = [fila.start, fila.start, fila.stop, fila.stop, fila.start]
        ax.plot(xs, ys, color=color, lw=2)
    # electrodo central: un círculo paramétrico (cos, sin) de radio R_elec.
    th = np.linspace(0, 2*np.pi, 100)          # 100 ángulos de 0 a 2π
    ax.plot(cy + R_elec*np.cos(th), cx + R_elec*np.sin(th), 'k', lw=2)

# ── Figura 1: potencial (3D + equipotenciales) ───────────────────────────────
fig1 = plt.figure(figsize=(14, 6))            # crea una figura vacía

# add_subplot(1, 2, 1, projection='3d') → 1 fila, 2 columnas, panel 1, en 3D.
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
# plot_surface dibuja V[i,j] como una superficie/relieve. cmap = mapa de colores.
ax1.plot_surface(X, Y, V, cmap='coolwarm', edgecolor='none')
ax1.set(xlabel='x', ylabel='y', zlabel='V [V]',
        title='Potencial V(x,y) - pico central = electrodo a 500 V')

ax2 = fig1.add_subplot(1, 2, 2)               # panel 2, en 2D
# contourf = mapa de color relleno por niveles; contour = sólo las líneas (equipotenciales).
cf = ax2.contourf(X, Y, V, levels=60, cmap='coolwarm')
ax2.contour(X, Y, V, levels=25, colors='k', linewidths=0.4, alpha=0.5)
dibujar_placas(ax2)
plt.colorbar(cf, ax=ax2, label='V [V]')       # barra de color que asocia color↔valor de V
ax2.set(xlabel='x', ylabel='y', title='Equipotenciales + placas y electrodo',
        aspect='equal')                        # aspect='equal' → no deforma el círculo
fig1.suptitle('ÍTEM 4 - Potencial: cuadrupolo + electrodo central (500 V)',
              fontsize=13)
plt.tight_layout()

# ── Figura 2: campo eléctrico + convergencia ─────────────────────────────────
fig2 = plt.figure(figsize=(14, 6))

ax3 = fig2.add_subplot(1, 2, 1)
ax3.contour(X, Y, V, levels=25, colors='gray', linewidths=0.4, alpha=0.5)   # equipotenciales de fondo
paso = 4                                       # "diezmado": dibujar 1 de cada 4 flechas (si no, se satura)
E_safe = np.where(E_mag > 0, E_mag, 1.0)       # evita dividir por 0 al normalizar las flechas
# quiver dibuja flechas (campo vectorial). Dibujamos la DIRECCIÓN (Ex/|E|, Ey/|E|)
# y coloreamos por la MAGNITUD |E|. [::paso, ::paso] toma 1 de cada 'paso' nodos.
ax3.quiver(X[::paso, ::paso], Y[::paso, ::paso],
           (Ex/E_safe)[::paso, ::paso], (Ey/E_safe)[::paso, ::paso],
           E_mag[::paso, ::paso], cmap='plasma', scale=35)
dibujar_placas(ax3)
ax3.set(xlabel='x', ylabel='y', title='ÍTEM 5 - Campo E = −∇V (sobre equipot.)',
        aspect='equal')

ax4 = fig2.add_subplot(1, 2, 2)
# .items() recorre el diccionario entregando (clave, valor) = (ω, historial).
for w, h in historiales.items():
    etiqueta = "GS (ω=1.0)" if w == 1.0 else f"SOR ω={w}"
    # semilogy = gráfica con eje Y logarítmico (ideal para ver el error caer órdenes de magnitud).
    ax4.semilogy(h, lw=2, label=f"{etiqueta}: {len(h)} iter")
ax4.axhline(tol, color='red', ls='--', lw=1, label=f'tol = {tol}')   # línea de la tolerancia
ax4.set(xlabel='Iteración', ylabel='Error máx. |ΔV|',
        title='ÍTEM 7 - Convergencia para distintos ω')
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)
plt.tight_layout()

# ── Comentarios físicos finales (ítems 4 y 5) ────────────────────────────────
# f"""...""" es una f-string de varias líneas: imprime el bloque con los valores ya sustituidos.
print("\n" + "═" * 64)
print("ÍTEMS 4 y 5 - Interpretación física")
print("═" * 64)
print(f"""  · Simetría: el potencial YA NO es antisimétrico ni nulo en las diagonales.
    El cuadrupolo solo daría V=0 en las diagonales, pero el electrodo central
    (simétrico, +500 V) añade un pedestal positivo: V(centro) = {V[cx,cy]:.0f} V.
  · Máximo de V en el electrodo (+500 V) y cerca de las placas +ρ₀;
    mínimo cerca de las placas −ρ₀ (V = {V.min():.0f} V).
  · Campo E: perpendicular a las equipotenciales en todo punto.  Cerca del
    electrodo apunta radialmente HACIA AFUERA (conductor positivo); muere
    en las placas −ρ₀ y en la pared aterrizada.  DENTRO del electrodo E ≈ 0
    (conductor en equilibrio: V constante ⇒ ∇V = 0).""")

plt.show()   # muestra todas las figuras (al final del programa)
