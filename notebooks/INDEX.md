# Índice de notebooks de clase

Notebooks de cada sesión del curso, organizados por unidad temática según el programa (ver [referencias/programa_fco4101.md](../referencias/programa_fco4101.md)). Cada unidad tiene su propio archivo de referencia resumido en `referencias/`.

## 00 — Fundamentos computacionales

| Notebook | Tema | Algoritmo clave |
|---|---|---|
| [clase_27-03-26.ipynb](00-fundamentos-computacionales/clase_27-03-26.ipynb) | Ecuación de Van der Pol (oscilador no lineal autoexcitado) | RK4, método de Euler |

## 01 — Dinámica clásica y no lineal ([referencias/cap03_dinamica_clasica_no_lineal.md](../referencias/cap03_dinamica_clasica_no_lineal.md))

| Notebook | Tema | Algoritmo clave |
|---|---|---|
| [clase_07-04-26.ipynb](01-dinamica-clasica-no-lineal/clase_07-04-26.ipynb) | Sistema de Rössler (atractor caótico 3D) | RK4, visualización 3D |
| [clase_09-04-26.ipynb](01-dinamica-clasica-no-lineal/clase_09-04-26.ipynb) | Órbitas: gravitación de dos cuerpos | RK4, conservación de energía |
| [clase_10-04-26.ipynb](01-dinamica-clasica-no-lineal/clase_10-04-26.ipynb) | Órbitas 2D (elípticas e hiperbólicas) | RK4, espacio de fase |
| [clase_28-04-26.ipynb](01-dinamica-clasica-no-lineal/clase_28-04-26.ipynb) | Péndulos acoplados / solitones anulares 2D | Leapfrog 2D |

## 02 — Ecuaciones de onda y dinámica de fluidos ([referencias/cap04_ecuaciones_onda_fluidos.md](../referencias/cap04_ecuaciones_onda_fluidos.md))

| Notebook | Tema | Algoritmo clave |
|---|---|---|
| [clase_14-04-26.ipynb](02-ecuaciones-onda-fluidos/clase_14-04-26.ipynb) | Ecuación de onda 1D, condiciones de frontera, estabilidad CFL | Leapfrog, diferencias finitas |
| [clase_16-04-26.ipynb](02-ecuaciones-onda-fluidos/clase_16-04-26.ipynb) | Modos normales: expansión analítica en serie de Fourier | Coeficientes de Fourier vs. solución numérica |
| [clase_21-04-26.ipynb](02-ecuaciones-onda-fluidos/clase_21-04-26.ipynb) | Cuerda real con tensión T(x) y densidad ρ(x) variables | Leapfrog con coeficientes interpolados en semipuntos |
| [clase_23-04-26.ipynb](02-ecuaciones-onda-fluidos/clase_23-04-26.ipynb) | Solitones en fibras ópticas (ecuación de Korteweg-de Vries) | Leapfrog, balance no lineal + dispersión |
| [clase_24-04-26.ipynb](02-ecuaciones-onda-fluidos/clase_24-04-26.ipynb) | Membrana 2D (onda bidimensional) | Leapfrog 2D con 4 vecinos |
| [clase_05-05-26.ipynb](02-ecuaciones-onda-fluidos/clase_05-05-26.ipynb) | Ecuación de Navier-Stokes (flujo incompresible) | RK4, energía del fluido |
| [clase_07-05-26.ipynb](02-ecuaciones-onda-fluidos/clase_07-05-26.ipynb) | Navier-Stokes con vorticidad (∇²ψ = -ω) | SOR iterativo |
| [clase_08-05-26.ipynb](02-ecuaciones-onda-fluidos/clase_08-05-26.ipynb) | Hidrodinámica: tanque de Torricelli | SOR, ley de Bernoulli |

## 03 — Electricidad y magnetismo ([referencias/cap05_electricidad_magnetismo.md](../referencias/cap05_electricidad_magnetismo.md))

| Notebook | Tema | Algoritmo clave |
|---|---|---|
| [clase_14-05-26_15-05-26.ipynb](03-electricidad-magnetismo/clase_14-05-26_15-05-26.ipynb) | Laplace/Poisson 2D: comparación Jacobi, Gauss-Seidel, SOR | SOR con ω óptimo |
| [clase_19-05-26.ipynb](03-electricidad-magnetismo/clase_19-05-26.ipynb) | Ondas electromagnéticas 1D (malla de Yee) | FDTD |
| [clase_21-05-26.ipynb](03-electricidad-magnetismo/clase_21-05-26.ipynb) | FDTD con medio dieléctrico (vacío-dieléctrico-vacío) | FDTD, análisis de dirección de propagación |
| [clase_28-05-26.ipynb](03-electricidad-magnetismo/clase_28-05-26.ipynb) | Cilindro dieléctrico en campo eléctrico uniforme | Laplace con ε(x,y) variable |

## 04 — Mecánica cuántica ([referencias/cap06_mecanica_cuantica.md](../referencias/cap06_mecanica_cuantica.md))

| Notebook | Tema | Algoritmo clave |
|---|---|---|
| [clase_02-06-26.ipynb](04-mecanica-cuantica/clase_02-06-26.ipynb) | Estados ligados en pozo cuadrado finito | Bisección sobre ecuación trascendental |
| [clase_05-06-26.ipynb](04-mecanica-cuantica/clase_05-06-26.ipynb) | Ecuación radial para átomo piónico de bario | RK4 (método de disparo) |
| [clase_09-06-26.ipynb](04-mecanica-cuantica/clase_09-06-26.ipynb) | Ecuación de Schrödinger dependiente del tiempo | Leapfrog con campo complejo (parte real/imaginaria) |
