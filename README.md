# Física Computacional (FCL1109) - Portafolio del curso

Repositorio del curso **FCL1109 (Física Computacional)**, Escuela de Física, Facultad de Ciencias Naturales y Matemática, Universidad de El Salvador, Ciclo I 2026. Libro de referencia principal: Landau & Páez, *Computational Problems for Physics* (CRC Press, 2018).

Este repositorio documenta el trabajo de tres estudiantes a lo largo de todo el curso: implementaciones de métodos numéricos, simulaciones de sistemas físicos, prácticas numéricas formales, exámenes parciales y proyectos finales de investigación. Se organiza como un portafolio técnico que demuestra experiencia práctica en **física computacional, métodos numéricos, simulación de sistemas dinámicos, ecuaciones diferenciales parciales, y métodos de Monte Carlo/cadenas de Markov**.

## Estudiantes

| Usuario de GitHub | Código universitario |
|---|---|
| [jofavalle](https://github.com/jofavalle) | AV18012 |
| [cesarp03](https://github.com/cesarp03) | PA22006 |
| [aalexanderrz](https://github.com/aalexanderrz) | RZ22004 |

## Programa del curso

| Semanas | Unidad | Referencia Landau | Material en este repo |
|---|---|---|---|
| 1–5 | Fundamentos computacionales y análisis de datos | Cap. 1–2 | [notebooks/00-fundamentos-computacionales](notebooks/00-fundamentos-computacionales/) |
| 6–8 | Dinámica clásica y no lineal (osciladores, caos, órbitas) | Cap. 3 | [notebooks/01-dinamica-clasica-no-lineal](notebooks/01-dinamica-clasica-no-lineal/), [referencias/cap03_dinamica_clasica_no_lineal.md](referencias/cap03_dinamica_clasica_no_lineal.md) |
| 9–13 | Ecuaciones de onda y dinámica de fluidos (FTCS, Crank-Nicolson, FFT) | Cap. 4 | [notebooks/02-ecuaciones-onda-fluidos](notebooks/02-ecuaciones-onda-fluidos/), [extras/plantillas-cod-unidad-4/](extras/plantillas-cod-unidad-4/), [referencias/cap04_ecuaciones_onda_fluidos.md](referencias/cap04_ecuaciones_onda_fluidos.md) |
| 14–15 | Electricidad y magnetismo (Laplace, Poisson, FDTD, SOR) | Cap. 5 | [notebooks/03-electricidad-magnetismo](notebooks/03-electricidad-magnetismo/), [referencias/cap05_electricidad_magnetismo.md](referencias/cap05_electricidad_magnetismo.md) |
| 16–17 | Mecánica cuántica | Cap. 6 | [notebooks/04-mecanica-cuantica](notebooks/04-mecanica-cuantica/), [referencias/cap06_mecanica_cuantica.md](referencias/cap06_mecanica_cuantica.md) |
| 18 | Termodinámica y física estadística | Cap. 7 | [parciales/parcial-3-...](parciales/parcial-3-examen-integrador-cuantica-electromagnetismo/) |

Evaluación: 4 prácticas numéricas (40 %) + 3 exámenes parciales (60 %). Programa completo en [referencias/programa_fco4101.md](referencias/programa_fco4101.md).

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| [seminarios-investigacion/](seminarios-investigacion/) | **Proyectos finales de investigación** (uno por estudiante) — la pieza más avanzada del portafolio: integrales de camino de Feynman con Monte Carlo/cadenas de Markov, caos en el mapa logístico, estados coherentes cuánticos |
| [practicas-numericas/](practicas-numericas/) | Las 3 prácticas numéricas formales del curso, resueltas por los 3 estudiantes |
| [parciales/](parciales/) | Exámenes parciales (onda 1D, examen integrador de cuántica y electromagnetismo) |
| [notebooks/](notebooks/) | Notebooks de cada clase, organizados por unidad temática ([índice](notebooks/INDEX.md)) |
| [scripts/](scripts/) | Scripts equivalentes a las clases, organizados por unidad temática ([índice](scripts/INDEX.md)) |
| [referencias/](referencias/) | Resúmenes ejecutivos de cada capítulo del libro guía, con los listings originales de Landau como autoridad algorítmica |
| [extras/](extras/) | Material de repaso y apoyo: exámenes simulacro, datasets (`datos_decaimiento.csv`, `senal_ruido.csv`) y [plantillas-cod-unidad-4/](extras/plantillas-cod-unidad-4/) (biblioteca de plantillas reutilizables para ondas y fluidos, fieles a los algoritmos de Landau) |
| [fcl1109.py](fcl1109.py) | Biblioteca numérica propia del curso (derivadas, integración, EDOs, ajuste de datos, raíces, Fourier) |

## Habilidades técnicas demostradas

- **Ecuaciones diferenciales ordinarias:** métodos de Euler y Runge-Kutta de 2º y 4º orden (RK2/RK4) aplicados a sistemas caóticos (Lorenz, Rössler), dinámica orbital, oscilador de Van der Pol y proyectiles con fricción.
- **Ecuaciones diferenciales parciales por diferencias finitas:** esquema leapfrog para la ecuación de onda 1D/2D (cuerdas, membranas, solitones KdV), FTCS y Crank-Nicolson para la ecuación de calor, SOR (sobrerrelajación sucesiva) para Laplace/Poisson y para Navier-Stokes en formulación de vorticidad-función de corriente.
- **Dinámica de fluidos compresibles:** ecuaciones de Euler 1D con el esquema de Lax-Friedrichs (ondas sonoras, efecto Doppler, ondas de choque).
- **Electromagnetismo computacional:** FDTD (malla de Yee) para propagación de ondas electromagnéticas en medios homogéneos y dieléctricos.
- **Mecánica cuántica computacional:** búsqueda de estados ligados (bisección, método de disparo), ecuación de Schrödinger dependiente del tiempo (leapfrog complejo), oscilador armónico cuántico, átomos piónicos.
- **Métodos de Monte Carlo y cadenas de Markov:** integración Monte Carlo, y el proyecto de investigación insignia que implementa integrales de camino de Feynman mediante Monte Carlo cuántico con el algoritmo de Metropolis (cadena de Markov con criterio de aceptación/rechazo).
- **Sistemas dinámicos y caos:** atractores extraños, diagramas de bifurcación, sensibilidad a condiciones iniciales, cascadas de duplicación de período.
- **Análisis numérico general:** diferenciación numérica, integración (trapecio, Simpson, cambio de variable para integrales impropias), ajuste por mínimos cuadrados, prueba de bondad de ajuste (χ²), transformada de Fourier discreta y FFT.

## Proyectos de investigación destacados

Los tres seminarios finales (ver [seminarios-investigacion/](seminarios-investigacion/)) son la evidencia más completa de dominio del curso:

- **jofavalle** — *Integral de camino de Feynman mediante Monte Carlo cuántico y cadenas de Markov (algoritmo de Metropolis)*: obtiene el estado fundamental del oscilador armónico cuántico muestreando trayectorias en tiempo imaginario, sin resolver la ecuación de Schrödinger.
- **cesarp03** — *Diagrama de bifurcación del mapa logístico*: transición de la dinámica estable a la caótica mediante iteración de mapas y análisis de bifurcaciones.
- **aalexanderrz** — *Estados coherentes de Glauber*: dinámica de superposiciones cuasi-clásicas del oscilador armónico cuántico mediante expansión en polinomios de Hermite.

## Entorno de ejecución

Entorno virtual de Python en `.venv/`. La biblioteca `fcl1109.py` (raíz del proyecto) centraliza los métodos numéricos usados en la mayoría de notebooks y scripts:

```bash
source .venv/bin/activate
python3 scripts/04-mecanica-cuantica/clase_09-06-26.py
```

Algunas prácticas de estudiantes incluyen su propia copia local de `fcl1109.py` (se conservan tal cual para no romper las entregas originales).
