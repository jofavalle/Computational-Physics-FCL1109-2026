# Plantillas de Código - Unidad 4: Ecuaciones de onda y dinámica de fluidos
> FCL1109 - Física Computacional, UES 2026  
> Referencia: Landau & Páez, *Computational Problems for Physics*, Cap. 4  
> ⚠ Solo se usan **numpy** y **matplotlib** (sin fcl1109, sin scipy)

---

## Estructura de un script típico

Todo script de la unidad 4 sigue este esqueleto (estudiar en orden):

```
1. Importaciones          → secciones/s01_importaciones.py
2. Parámetros + grilla    → secciones/s02_parametros_grilla.py
3. Condiciones iniciales  → secciones/s03_condiciones_iniciales.py
4. Condiciones de frontera→ secciones/s04_condiciones_frontera.py
5. Loop de integración    → secciones/s05_loops_integracion.py
6. Postprocesamiento      → secciones/s07_postprocesamiento.py
7. Visualización          → secciones/s06_visualizacion.py   ← CLAVE para el parcial
```

### Plantillas de secciones (`secciones/`)

| Archivo | Contenido |
|---------|-----------|
| `s01_importaciones.py` | numpy, matplotlib, mpl_toolkits, FuncAnimation |
| `s02_parametros_grilla.py` | Onda 1D, onda 2D, calor, N-S - con assert de estabilidad |
| `s03_condiciones_iniciales.py` | Gaussiana, triángulo, seno, escalón, ruido, 2D |
| `s04_condiciones_frontera.py` | Dirichlet, Neumann, periódica, máscara obstáculos |
| `s05_loops_integracion.py` | Leapfrog 1D/2D, FTCS, SOR Laplace, SOR N-S, np.roll |
| `s06_visualizacion.py` | **plot, imshow, contourf, streamplot, quiver, stem, 3D, animación** |
| `s07_postprocesamiento.py` | v desde ψ, E desde V, energía, FFT, error de convergencia |

---

## Mapa de plantillas completas

| Archivo | Tema | Ecuación central |
|---------|------|-----------------|
| `01_onda_1d.py` | Ecuación de onda 1D | `y[i,n+1] = 2y[i,n] - y[i,n-1] + r²(y[i+1,n] - 2y[i,n] + y[i-1,n])` |
| `02_modos_normales.py` | Modos normales + FFT | `y = Σ Bₙ sin(nπx/L) cos(ωₙt)` |
| `03_onda_propiedades_variables.py` | Cuerda con T(x), ρ(x) | Laplaciano con T interpolada en semipuntos |
| `04_membrana_2d.py` | Membrana vibrante 2D | Onda 2D con vecinos `i±1, j±1` |
| `05_burgers.py` | Ecuación de Burgers | `u[i] + ε*laplaciano - μ*u*∂u/∂x` |
| `06_calor_ftcs.py` | Ecuación de calor FTCS | `T[i] + r*(T[i+1] - 2T[i] + T[i-1])` |
| `07_laplace_poisson_2d.py` | Laplace / Poisson 2D | `V[i,j] = 0.25*(suma vecinos) + término ρ` |
| `08_navier_stokes_basico.py` | N-S: ψ-ω canal simple | SOR para ψ y ω acopladas |
| `09_navier_stokes_viga.py` | N-S: canal con viga | N-S + saltar interior de la viga |
| `10_navier_stokes_agujero.py` | N-S: boquilla/agujero | N-S + cc especiales en el agujero |
| `11_fft_espectral.py` | FFT y análisis espectral | `np.fft.fft`, filtrado paso-bajos |

---

## Condiciones de estabilidad (¡imprescindibles en el parcial!)

| Esquema | Condición | Fórmula |
|---------|-----------|---------|
| **Onda 1D** (CFL) | `r ≤ 1` | `r = c·dt/dx` |
| **Onda 2D** (CFL 2D) | `r ≤ 1/√2 ≈ 0.707` | `r = c·dt/dx` |
| **Calor FTCS** (Von Neumann) | `r ≤ 0.5` | `r = α·dt/dx²` |
| **N-S / SOR** | iterativo, `ω ∈ (0,2)` | `ω ≈ 0.1-1.9` |
| **Crank-Nicolson** | incondicionalmente estable | no se implementa sin scipy |

> **Regla rápida para dt:**  
> - Onda 1D: `dt = 0.8 * dx / c`  
> - Onda 2D: `dt = 0.6 * dx / c`  
> - Calor:   `dt = 0.4 * dx² / alpha`

---

## Fórmulas de discretización clave

### Onda 1D
```
y[i,n+1] = 2*y[i,n] - y[i,n-1] + r²*(y[i+1,n] - 2*y[i,n] + y[i-1,n])
```
Con amortiguamiento κ:
```
y_nuevo[i] = (2*y[i] - y_old[i] + r²*(laplaciano) + κ·dt·y_old[i]) / (1 + κ·dt)
```

### Onda con T(x), ρ(x) variables
```python
T_ip = 0.5*(T[i] + T[i+1])          # T en i+1/2
T_im = 0.5*(T[i] + T[i-1])          # T en i-1/2
lap  = (T_ip*(y[i+1]-y[i]) - T_im*(y[i]-y[i-1])) / dx²
y_nuevo[i] = 2*y[i] - y_old[i] + dt²*lap/rho[i]
```

### Calor FTCS
```
T[i,n+1] = T[i,n] + r*(T[i+1,n] - 2*T[i,n] + T[i-1,n])
```

### Navier-Stokes ψ-ω (SOR)
```python
# Poisson para ψ:
psi_nuevo = 0.25*(psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] + h²*w[i,j])
psi[i,j] += omega*(psi_nuevo - psi[i,j])

# Transporte de vorticidad:
w_nuevo = 0.25*(w[i+1,j] + w[i-1,j] + w[i,j+1] + w[i,j-1])
         + (R/16)*[(psi[j+1]-psi[j-1])*(w[i+1]-w[i-1]) - (psi[i+1]-psi[i-1])*(w[j+1]-w[j-1])]
w[i,j] += omega*(w_nuevo - w[i,j])
```

### Jacobi / Gauss-Seidel para Laplace
```python
V[i,j] = 0.25*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])   # Gauss-Seidel (in-place)
# Poisson: agregar + 0.25*dx²*rho[i,j]/eps0
```

### FFT con numpy
```python
U     = np.fft.fft(u)
freqs = np.fft.fftfreq(N, d=dx)
amp   = np.abs(U) / N * 2     # mitad positiva
# IFFT:
u_rec = np.fft.ifft(U).real
```

---

## Condiciones iniciales típicas

```python
# Pulso gaussiano
u0 = np.exp(-((x - x0)**2) / (2*sigma**2))

# Cuerda triangular
u0 = np.where(x < L/2, 2*x/L, 2*(L-x)/L)

# Modo n=1
u0 = np.sin(np.pi * x / L)

# Escalón suavizado (Burgers)
u0 = 0.5*(1 - np.tanh(x/5 - 5))
```

---

## Condiciones de frontera

| Tipo | Código | Uso |
|------|--------|-----|
| Dirichlet (valor fijo) | `u[0] = 0; u[-1] = 0` | cuerda fija, temperatura fija |
| Neumann (derivada cero) | `u[0] = u[1]; u[-1] = u[-2]` | extremo libre, aislado |
| Periódica | `u[0] = u[-2]; u[-1] = u[1]` | señal periódica |

---

## Errores frecuentes

| Error | Causa | Solución |
|-------|-------|---------|
| Solución explota/oscila | CFL violado (r > límite) | Reducir `dt` o aumentar `dx` |
| N-S no converge | `ω` demasiado grande | Reducir `omega` (empezar con 0.1) |
| FFT con picos falsos | Señal no periódica | Ventana de Hanning: `u *= np.hanning(N)` |
| Laplace converge lento | ω demasiado pequeño | Aumentar hasta ~1.9 (SOR óptimo) |
| Membrana inestable | CFL 2D: `r > 1/√2` | `dt = 0.6*dx/c` |

---

## Número de Reynolds

```
R = V₀ * L / ν
R < 2300  → flujo laminar
R > 4000  → flujo turbulento
```
En la grilla discreta se usa: `R = V₀ * h / ν`

---

## Modelos de cuerda (plantilla 03)

| Modelo | T(x) | ρ(x) |
|--------|------|------|
| Exponencial | `T₀·exp(αx)` | `ρ₀·exp(αx)` |
| Catenaria | `T₀·cosh(ρ₀gx/T₀)` | `ρ₀` (uniforme) |

Velocidad local: `v(x) = sqrt(T(x)/ρ(x))`  
Paso temporal: `dt = 0.4 * dx / max(v)`
