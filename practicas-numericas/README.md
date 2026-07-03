# Prácticas numéricas

Las 3 prácticas numéricas formales del curso FCL1109 (40 % de la nota final). Cada práctica agrupa las entregas de los 3 estudiantes: [jofavalle](https://github.com/jofavalle) (AV18012), [cesarp03](https://github.com/cesarp03) (PA22006) y [aalexanderrz](https://github.com/aalexanderrz) (RZ22004), cada uno en su propia subcarpeta.

## Práctica 1 - [Fundamentos numéricos: derivadas, integrales y EDOs clásicas](practica-1-fundamentos-numericos-edo-integrales/)

| Estudiante | Problema | Tema físico | Técnica numérica |
|---|---|---|---|
| jofavalle | `legendre_rodrigues_derivadas_numericas.py` | Polinomios de Legendre (fórmula de Rodrigues) | Derivadas numéricas por diferencias centrales |
| jofavalle | `debye_fotones_integrales_temperaturas.py` | Energía de un gas de fotones (modelo de Debye) | Integración numérica (trapecio, Simpson) |
| jofavalle | `lorenz_atractor_rk2.py` | Atractor de Lorenz | RK2 (punto medio) |
| jofavalle | `proyectil_friccion_rk4.py` | Proyectil con resistencia del aire | RK4 |
| cesarp03 | `legendre_rodrigues_derivadas_numericas.py` | Polinomios de Legendre (fórmula de Rodrigues) | Derivadas numéricas por diferencias centrales |
| cesarp03 | `debye_fotones_integrales.py` | Energía de un gas de fotones (modelo de Debye) | Integración con cambio de variable para integrales impropias |
| cesarp03 | `lorenz_atractor_rk2.py` | Atractor de Lorenz | RK2 (punto medio) |
| cesarp03 | `proyectil_friccion_rk4.py` | Proyectil con resistencia del aire (3 leyes de fricción) | RK4 |
| cesarp03 | `edo_acopladas_2d_rk4.py`, `utils_integracion_numerica.py`, `utils_transformacion_variable.py` | Utilidades de apoyo (EDOs acopladas, integración, cambio de variable) | RK4 / Monte Carlo / Simpson |
| aalexanderrz | `legendre_derivadas_numericas.py` | Polinomios de Legendre | Derivadas numéricas por diferencias centrales |
| aalexanderrz | `lorenz_atractor_rk2.py` | Atractor de Lorenz | RK2 (punto medio) |
| aalexanderrz | `proyectil_resistencia_aire_rk4.py` | Proyectil con resistencia del aire | RK4 |

> **Hueco conocido:** el problema de energía de Debye de aalexanderrz (equivalente al `debye_fotones_integrales.py` de sus compañeros) no está disponible en el repositorio.

## Práctica [Dispersión elástica en un potencial 2D](practica-2-dispersion-potencial-2d-rk4/)

Los 3 estudiantes resuelven la trayectoria de dispersión de una partícula incidente sobre un potencial central 2D integrando las ecuaciones de movimiento con RK4, y analizan el ángulo de dispersión en función del parámetro de impacto.

| Estudiante | Archivo |
|---|---|
| jofavalle | [dispersion_potencial_2d_rk4.ipynb](practica-2-dispersion-potencial-2d-rk4/jofavalle/dispersion_potencial_2d_rk4.ipynb) |
| cesarp03 | [dispersion_potencial_2d_rk4.ipynb](practica-2-dispersion-potencial-2d-rk4/cesarp03/dispersion_potencial_2d_rk4.ipynb) |
| aalexanderrz | [dispersion_potencial_2d_rk4.ipynb](practica-2-dispersion-potencial-2d-rk4/aalexanderrz/dispersion_potencial_2d_rk4.ipynb) |

## Práctica 3 - [Ecuaciones de Euler para un fluido compresible](practica-3-euler-fluido-compresible-lax-friedrichs/)

Resolución del sistema de Euler 1D (continuidad, momento, energía) en forma conservativa con el esquema de Lax-Friedrichs, estudiando fenómenos como ondas sonoras, efecto Doppler, propagación bidireccional, conservación de integrales, convergencia numérica, estabilidad (CFL) y régimen no lineal (ondas de choque).

| Estudiante | Archivo(s) |
|---|---|
| jofavalle | [euler_fluido_compresible_lax_friedrichs.ipynb](practica-3-euler-fluido-compresible-lax-friedrichs/jofavalle/euler_fluido_compresible_lax_friedrichs.ipynb) |
| cesarp03 | [euler_fluido_compresible_lax_friedrichs.ipynb](practica-3-euler-fluido-compresible-lax-friedrichs/cesarp03/euler_fluido_compresible_lax_friedrichs.ipynb) |
| aalexanderrz | 7 scripts (`euler_ondas_sonoras`, `euler_efecto_doppler`, `euler_onda_bidireccional`, `euler_conservacion_integrales`, `euler_convergencia_numerica`, `euler_estabilidad_cfl`, `euler_regimen_no_lineal`), cada uno estudiando un fenómeno distinto con el mismo esquema |
