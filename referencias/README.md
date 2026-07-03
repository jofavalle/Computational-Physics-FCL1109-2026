# Referencias - Física Computacional FCO4101
> Universidad de El Salvador · Ciclo I 2026  
> Basado en: Landau & Páez, *Computational Problems for Physics*, CRC Press 2018

---

## Archivos de referencia

| Archivo | Capítulo | Temas |
|---------|----------|-------|
| [`cap03_dinamica_clasica_no_lineal.md`](cap03_dinamica_clasica_no_lineal.md) | Cap. 3 | Oscilador, péndulo, caos, Poincaré, mapa logístico, RK4 |
| [`cap04_ecuaciones_onda_fluidos.md`](cap04_ecuaciones_onda_fluidos.md) | Cap. 4 | Onda 1D, calor, Laplace, Poisson, FTCS, Crank-Nicolson, FFT |
| [`cap05_electricidad_magnetismo.md`](cap05_electricidad_magnetismo.md) | Cap. 5 | Potencial, relajación, Biot-Savart, FDTD, capacitancia |
| [`cap06_mecanica_cuantica.md`](cap06_mecanica_cuantica.md) | Cap. 6 | Estados ligados, Numerov, Schrödinger t-dependiente, dispersión, QM matricial, qubits, Feynman |

---

## Métodos numéricos - resumen rápido

| Método | Cuándo usarlo | Orden de error |
|--------|--------------|---------------|
| RK4 | EDOs, dinámica | O(h⁴) |
| Euler explícito | Solo para aprender | O(h) |
| FTCS | Difusión (dt pequeño) | O(dt, dx²) |
| Crank-Nicolson | Difusión estable | O(dt², dx²) |
| Diferencias finitas onda | Ecuación de onda | O(dt², dx²) |
| Jacobi/Gauss-Seidel/SOR | Laplace, Poisson | Iterativo |
| FDTD (Yee) | Maxwell en el tiempo | O(dt², dx²) |
| Numerov | EDO sin 1ª derivada (Schrödinger) | O(h⁶) |
| Leapfrog Schrödinger (R/I escalonados) | Schrödinger dependiente del tiempo | O(dt², dx²) |
| Cuadratura de Gauss + autovalores | QM en espacio de momentos (ec. integral) | Espectral |
| Metropolis | Integral de camino de Feynman, estado base | Monte Carlo |

---

## Criterios de estabilidad (resumen)

```
Onda:     r = c·Δt/Δx  ≤ 1          (Courant-Friedrichs-Lewy)
Calor:    α·Δt/Δx²     ≤ 0.5        (Von Neumann)
FDTD:     c·Δt/Δx      ≤ 1/√d      (d = dimensiones espaciales)
```
