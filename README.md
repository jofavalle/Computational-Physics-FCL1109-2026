# Computational Physics (FCL1109) - Course Portfolio

English | [Español](README.es.md)

Repository for the course **FCL1109 (Computational Physics)**, School of Physics, Faculty of Natural Sciences and Mathematics, University of El Salvador (Universidad de El Salvador), first semester of 2026. Main textbook: Landau & Páez, *Computational Problems for Physics* (CRC Press, 2018).

This repository documents the work of three students throughout the whole course: implementations of numerical methods, simulations of physical systems, formal numerical assignments, midterm exams and final research projects. It is organized as a technical portfolio of hands-on work in **computational physics, numerical methods, simulation of dynamical systems, partial differential equations, and Monte Carlo/Markov chain methods**.

Course materials (code comments, notebooks and reports) are in Spanish, the language of instruction at the University of El Salvador.

## Students

| GitHub user | Student ID |
|---|---|
| [jofavalle](https://github.com/jofavalle) | AV18012 |
| [cesarp03](https://github.com/cesarp03) | PA22006 |
| [aalexanderrz](https://github.com/aalexanderrz) | RZ22004 |

## Course syllabus

| Weeks | Unit | Landau chapter | Material in this repository |
|---|---|---|---|
| 1-5 | Computational fundamentals and data analysis | Ch. 1-2 | [notebooks/00-fundamentos-computacionales](notebooks/00-fundamentos-computacionales/) |
| 6-8 | Classical and nonlinear dynamics (oscillators, chaos, orbits) | Ch. 3 | [notebooks/01-dinamica-clasica-no-lineal](notebooks/01-dinamica-clasica-no-lineal/), [referencias/cap03_dinamica_clasica_no_lineal.md](referencias/cap03_dinamica_clasica_no_lineal.md) |
| 9-13 | Wave equations and fluid dynamics (FTCS, Crank-Nicolson, FFT) | Ch. 4 | [notebooks/02-ecuaciones-onda-fluidos](notebooks/02-ecuaciones-onda-fluidos/), [extras/plantillas-cod-unidad-4/](extras/plantillas-cod-unidad-4/), [referencias/cap04_ecuaciones_onda_fluidos.md](referencias/cap04_ecuaciones_onda_fluidos.md) |
| 14-15 | Electricity and magnetism (Laplace, Poisson, FDTD, SOR) | Ch. 5 | [notebooks/03-electricidad-magnetismo](notebooks/03-electricidad-magnetismo/), [referencias/cap05_electricidad_magnetismo.md](referencias/cap05_electricidad_magnetismo.md) |
| 16-17 | Quantum mechanics | Ch. 6 | [notebooks/04-mecanica-cuantica](notebooks/04-mecanica-cuantica/), [referencias/cap06_mecanica_cuantica.md](referencias/cap06_mecanica_cuantica.md) |
| 18 | Thermodynamics and statistical physics | Ch. 7 | [parciales/parcial-3-...](parciales/parcial-3-examen-integrador-cuantica-electromagnetismo/) |

Assessment: 4 numerical assignments (40%) and 3 midterm exams (60%). Full syllabus (in Spanish) in [referencias/programa_fcl1109.md](referencias/programa_fcl1109.md).

## Repository structure

| Folder | Contents |
|---|---|
| [seminarios-investigacion/](seminarios-investigacion/) | **Final research projects** (one per student), the most advanced work in the portfolio: Feynman path integrals with Monte Carlo/Markov chains, chaos in the logistic map, quantum coherent states |
| [practicas-numericas/](practicas-numericas/) | The 3 formal numerical assignments of the course, solved by each of the 3 students |
| [parciales/](parciales/) | Midterm exams (1D wave equation; comprehensive exam on quantum mechanics and electromagnetism) |
| [notebooks/](notebooks/) | Notebooks from each lecture, organized by unit ([index](notebooks/INDEX.md)) |
| [scripts/](scripts/) | Scripts matching the lectures, organized by unit ([index](scripts/INDEX.md)) |
| [referencias/](referencias/) | Summaries of chapters 3 to 6 of the textbook and the full course syllabus |
| [extras/](extras/) | Review and support material: practice exams, datasets (`datos_decaimiento.csv`, `senal_ruido.csv`) and [plantillas-cod-unidad-4/](extras/plantillas-cod-unidad-4/) (a library of reusable templates for waves and fluids that follow Landau's algorithms) |
| [fcl1109.py](fcl1109.py) | The course's own numerical library (derivatives, integration, ODEs, data fitting, root finding, Fourier analysis) |

## Technical skills demonstrated

- **Ordinary differential equations:** Euler and second- and fourth-order Runge-Kutta methods (RK2/RK4) applied to chaotic systems (Lorenz, Rössler), orbital dynamics, the Van der Pol oscillator and projectiles with drag.
- **Partial differential equations by finite differences:** leapfrog scheme for the 1D/2D wave equation (strings, membranes, KdV solitons), FTCS and Crank-Nicolson for the heat equation, and SOR (successive over-relaxation) for Laplace/Poisson and for Navier-Stokes in the vorticity-stream function formulation.
- **Compressible fluid dynamics:** 1D Euler equations with the Lax-Friedrichs scheme (sound waves, Doppler effect, shock waves).
- **Computational electromagnetism:** FDTD (Yee lattice) for electromagnetic wave propagation in homogeneous and dielectric media.
- **Computational quantum mechanics:** bound-state search (bisection, shooting method), the time-dependent Schrödinger equation (complex leapfrog), the quantum harmonic oscillator and pionic atoms.
- **Monte Carlo methods and Markov chains:** Monte Carlo integration, and the research project that computes Feynman path integrals through quantum Monte Carlo with the Metropolis algorithm (a Markov chain with an acceptance/rejection criterion).
- **Dynamical systems and chaos:** strange attractors, bifurcation diagrams, sensitivity to initial conditions and period-doubling cascades.
- **General numerical analysis:** numerical differentiation, integration (trapezoidal rule, Simpson's rule, change of variables for improper integrals), least-squares fitting, chi-squared goodness-of-fit test (χ²), discrete Fourier transform and FFT.

## Featured research projects

The three final seminars (see [seminarios-investigacion/](seminarios-investigacion/)) are the most complete work of the course:

- **jofavalle** - *Feynman path integral through quantum Monte Carlo and Markov chains (Metropolis algorithm)*: obtains the ground state of the quantum harmonic oscillator by sampling imaginary-time paths, without solving the Schrödinger equation.
- **cesarp03** - *Bifurcation diagram of the logistic map*: the transition from stable to chaotic dynamics, studied through map iteration and bifurcation analysis.
- **aalexanderrz** - *Glauber coherent states*: dynamics of quasi-classical superpositions of the quantum harmonic oscillator through an expansion in Hermite polynomials.

## Running the code

Python virtual environment in `.venv/`. The library `fcl1109.py` (repository root) gathers the numerical methods used by most notebooks and scripts:

```bash
source .venv/bin/activate
python3 scripts/04-mecanica-cuantica/clase_09-06-26.py
```

Some student assignments include their own local copy of `fcl1109.py`, kept as-is so that the original submissions still run.
