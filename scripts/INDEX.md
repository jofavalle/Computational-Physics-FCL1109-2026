# Índice de scripts de clase

Scripts `.py` correspondientes a cada sesión (complementos o variantes de los notebooks), organizados por unidad temática. `fcl1109.py` (raíz del repositorio) es la biblioteca numérica compartida usada por estos scripts.

## 00 - Fundamentos computacionales

| Script | Tema | Algoritmo clave |
|---|---|---|
| [clase_28-05-26_2.py](00-fundamentos-computacionales/clase_28-05-26_2.py) | Integral elíptica completa de primera especie K(m) y ajuste de una aproximación racional-logarítmica | Simpson (`fc.simpson`), mínimos cuadrados |

## 01 - Dinámica clásica y no lineal

| Script | Tema | Algoritmo clave |
|---|---|---|
| [clase_10-04-26.py](01-dinamica-clasica-no-lineal/clase_10-04-26.py) | Órbitas 2D bajo potencial central | RK4 |

## 02 - Ecuaciones de onda y dinámica de fluidos

| Script | Tema | Algoritmo clave |
|---|---|---|
| [clase_05-05-26.py](02-ecuaciones-onda-fluidos/clase_05-05-26.py) | Navier-Stokes básico (condiciones de frontera) | SOR para ψ y ω |
| [clase_07-05-26.py](02-ecuaciones-onda-fluidos/clase_07-05-26.py) | Navier-Stokes con vorticidad, canal con viga interior | SOR ψ + vorticidad con término convectivo |
| [clase_08-05-26.py](02-ecuaciones-onda-fluidos/clase_08-05-26.py) | Navier-Stokes con agujero de drenaje (Torricelli) | SOR + condiciones especiales de Bernoulli |
| [clase_21-04-26.py](02-ecuaciones-onda-fluidos/clase_21-04-26.py) | Cuerda con tensión/densidad variable | Leapfrog con T(x) interpolada |
| [clase_23-04-26.py](02-ecuaciones-onda-fluidos/clase_23-04-26.py) | Membrana vibrante 2D | Leapfrog 2D |
| [clase_24-04-26_animated.py](02-ecuaciones-onda-fluidos/clase_24-04-26_animated.py) | Membrana 2D animada | Leapfrog 2D + `FuncAnimation` |
| [clase_24-04-26_surface_3d.py](02-ecuaciones-onda-fluidos/clase_24-04-26_surface_3d.py) | Membrana 2D, visualización de superficie 3D | `plot_surface` |
| [clase_24-04-26_velocity_analysis.py](02-ecuaciones-onda-fluidos/clase_24-04-26_velocity_analysis.py) | Análisis de velocidades de la membrana | Derivada temporal del desplazamiento |

## 03 - Electricidad y magnetismo

| Script | Tema | Algoritmo clave |
|---|---|---|
| [clase_19-05-26.py](03-electricidad-magnetismo/clase_19-05-26.py) | Ondas electromagnéticas 1D, pulso gaussiano | FDTD (malla de Yee) |
| [clase_21-05-26.py](03-electricidad-magnetismo/clase_21-05-26.py) | FDTD con medio dieléctrico y análisis de dirección de propagación (vector de Poynting) | FDTD |
| [clase_28-05-26.py](03-electricidad-magnetismo/clase_28-05-26.py) | Cilindro dieléctrico en campo eléctrico uniforme | Relajación / diferencias finitas 2D |

## 04 - Mecánica cuántica

| Script | Tema | Algoritmo clave |
|---|---|---|
| [clase_04-06-26.py](04-mecanica-cuantica/clase_04-06-26.py) | Estados ligados: búsqueda de autovalores de energía | RK4 + bisección (`QuantumEigenCall`) |
| [clase_05-06-26.py](04-mecanica-cuantica/clase_05-06-26.py) | Ecuación radial relativista para átomo piónico de bario | RK4 (shooting) |
| [clase_05-06-26_v2.py](04-mecanica-cuantica/clase_05-06-26_v2.py) | Variante del script anterior (átomo piónico) | RK4 (shooting) |
| [clase_09-06-26.py](04-mecanica-cuantica/clase_09-06-26.py) | Schrödinger dependiente del tiempo | Leapfrog con campo complejo |
