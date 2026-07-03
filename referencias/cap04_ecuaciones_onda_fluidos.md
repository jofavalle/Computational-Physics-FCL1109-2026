# Capítulo 4 - Ecuaciones de onda y dinámica de fluidos
> Referencia: Landau & Páez, *Computational Problems for Physics* (2018), Cap. 4  
> Curso: FCL1109 - Física Computacional, UES 2026

---

## 4.1 Ecuación de onda clásica (1D)

### Forma diferencial

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

donde `c` es la velocidad de propagación.

### Discretización en diferencias finitas

Con grilla espacial `Δx` y paso temporal `Δt`:

$$u_i^{n+1} = 2u_i^n - u_i^{n-1} + \left(\frac{c\,\Delta t}{\Delta x}\right)^2 \!\left(u_{i+1}^n - 2u_i^n + u_{i-1}^n\right)$$

> ⚠️ **Condición de estabilidad de Courant (CFL):** el esquema es estable solo si:
> $$r = \frac{c\,\Delta t}{\Delta x} \leq 1$$
> Landau usa típicamente `r = 1.0` exacto para máxima precisión.

```python
def avanzar_onda(u_actual, u_anterior, c, dx, dt):
    r = c * dt / dx
    assert r <= 1.0, f"Inestable: r = {r:.3f} > 1"
    u_nuevo = np.zeros_like(u_actual)
    # Nodos interiores
    u_nuevo[1:-1] = (2*u_actual[1:-1] - u_anterior[1:-1]
                     + r**2 * (u_actual[2:] - 2*u_actual[1:-1] + u_actual[:-2]))
    # Condiciones de frontera (Dirichlet: u=0 en extremos)
    u_nuevo[0] = 0.0
    u_nuevo[-1] = 0.0
    return u_nuevo
```

---

## 4.2 Condiciones de frontera y condiciones iniciales

### Tipos de condiciones de frontera

| Tipo | Descripción | Implementación |
|------|-------------|----------------|
| Dirichlet | Valor fijo en la frontera | `u[0] = u[-1] = 0` |
| Neumann | Derivada fija (reflexión) | `u[0] = u[1]` |
| Periódica | `u[0] = u[-1]` | índices módulo N |

### Condiciones iniciales típicas (Landau Cap. 4)

**Pulso gaussiano:**
```python
x = np.linspace(0, L, N)
sigma = L / 10
x0 = L / 2
u0 = np.exp(-((x - x0)**2) / (2*sigma**2))
u_dot0 = np.zeros(N)  # velocidad inicial cero
# Primer paso con velocidad inicial:
u1 = u0 + dt * u_dot0
```

**Cuerda pulsada (triángulo):**
```python
u0 = np.where(x < L/2, 2*x/L, 2*(L-x)/L)
```

---

## 4.3 Ecuación de calor (difusión)

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

donde `α` es la difusividad térmica.

### Esquema explícito (FTCS - Forward Time, Centered Space)

$$T_i^{n+1} = T_i^n + \alpha \frac{\Delta t}{(\Delta x)^2}\left(T_{i+1}^n - 2T_i^n + T_{i-1}^n\right)$$

> ⚠️ **Condición de estabilidad:**
> $$\alpha \frac{\Delta t}{(\Delta x)^2} \leq \frac{1}{2}$$

```python
def paso_difusion_ftcs(T, alpha, dx, dt):
    r = alpha * dt / dx**2
    assert r <= 0.5, f"Inestable: r = {r:.4f}"
    T_nuevo = T.copy()
    T_nuevo[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    return T_nuevo
```

### Esquema implícito de Crank-Nicolson (más estable)

$$-r T_{i-1}^{n+1} + (2+2r)T_i^{n+1} - r T_{i+1}^{n+1} = r T_{i-1}^n + (2-2r)T_i^n + r T_{i+1}^n$$

```python
from scipy.linalg import solve_banded

def paso_crank_nicolson(T, alpha, dx, dt):
    r = alpha * dt / dx**2
    N = len(T)
    # Construir sistema tridiagonal
    diag_principal = np.full(N, 2 + 2*r)
    diag_sup = np.full(N-1, -r)
    diag_inf = np.full(N-1, -r)
    # Lado derecho
    b = r*T[:-2] + (2-2*r)*T[1:-1] + r*T[2:]
    # Resolver (condiciones Dirichlet en extremos)
    # ... (implementación con scipy.linalg.solve_banded)
    return T_nuevo
```

---

## 4.4 Ecuación de Laplace y Poisson (estado estacionario)

### Ecuación de Laplace (sin fuentes)

$$\nabla^2 \phi = \frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} = 0$$

### Método de relajación (iteración de Jacobi)

En cada punto interior de la grilla 2D:

$$\phi_{i,j}^{(n+1)} = \frac{1}{4}\left(\phi_{i+1,j}^{(n)} + \phi_{i-1,j}^{(n)} + \phi_{i,j+1}^{(n)} + \phi_{i,j-1}^{(n)}\right)$$

```python
def relajacion_jacobi(phi, tolerancia=1e-5, max_iter=10000):
    for iteracion in range(max_iter):
        phi_nuevo = phi.copy()
        phi_nuevo[1:-1, 1:-1] = 0.25 * (
            phi[2:, 1:-1] + phi[:-2, 1:-1] +
            phi[1:-1, 2:] + phi[1:-1, :-2]
        )
        # Aplicar condiciones de frontera
        # ...
        error = np.max(np.abs(phi_nuevo - phi))
        phi = phi_nuevo
        if error < tolerancia:
            print(f"Convergió en {iteracion} iteraciones")
            break
    return phi
```

### Ecuación de Poisson (con fuente ρ)

$$\nabla^2 \phi = -\frac{\rho}{\epsilon_0}$$

```python
# Modificación de Jacobi para Poisson:
phi_nuevo[1:-1, 1:-1] = 0.25 * (
    phi[2:, 1:-1] + phi[:-2, 1:-1] +
    phi[1:-1, 2:] + phi[1:-1, :-2]
    + dx**2 * rho[1:-1, 1:-1] / epsilon_0
)
```

---

## 4.5 Dinámica de fluidos - Ecuaciones de Navier-Stokes (simplificadas)

### Ecuación de continuidad (fluido incompresible)

$$\nabla \cdot \mathbf{v} = 0$$

### Ecuación de momentum (forma simplificada 1D con viscosidad)

$$\frac{\partial v}{\partial t} = -v\frac{\partial v}{\partial x} + \nu\frac{\partial^2 v}{\partial x^2}$$

donde `ν` es la viscosidad cinemática.

### Número de Reynolds

$$Re = \frac{v_0 L}{\nu}$$

- `Re < 2300`: flujo laminar
- `Re > 4000`: flujo turbulento

---

## 4.6 Transformada de Fourier (análisis espectral de ondas)

```python
import numpy as np

def analisis_espectral(u, dx):
    N = len(u)
    U = np.fft.fft(u)
    freqs = np.fft.fftfreq(N, d=dx)
    amplitudes = np.abs(U) / N
    return freqs[:N//2], amplitudes[:N//2]

# Uso típico:
freqs, amp = analisis_espectral(u_t, dx)
plt.plot(freqs, amp)
plt.xlabel('Frecuencia espacial (1/m)')
plt.ylabel('Amplitud')
```

---

## 4.7 Configuración de grilla numérica

```python
# Grilla estándar para problemas de onda / difusión 1D
def crear_grilla_1d(L, N, t_max, c=1.0, factor_cfl=0.9):
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    dt = factor_cfl * dx / c   # cumple CFL automáticamente
    t = np.arange(0, t_max + dt, dt)
    return x, dx, t, dt

# Grilla 2D para Laplace/Poisson
def crear_grilla_2d(Lx, Ly, Nx, Ny):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    phi = np.zeros((Nx, Ny))
    return x, y, dx, dy, phi
```

---

## Problemas típicos del libro (Cap. 4)

- **4.1** Cuerda vibrante: propagar pulso gaussiano, verificar reflexión en extremos
- **4.2** Ecuación de calor: barra con extremos a temperaturas fijas, comparar FTCS vs Crank-Nicolson
- **4.3** Laplace 2D: caja conductora con una pared a potencial V₀, trazar equipotenciales
- **4.4** Poisson: distribución de carga puntual, comparar con solución analítica 1/r
- **4.5** Análisis de modos normales con FFT de la cuerda en estado estacionario

---

## Advertencias frecuentes

| Error | Causa | Solución |
|---|---|---|
| Solución explota | CFL violado | Reducir dt o aumentar dx |
| Convergencia lenta en Jacobi | Grilla muy fina | Usar SOR (ω ≈ 1.9) o método de Gauss-Seidel |
| FFT con artefactos | Señal no periódica | Aplicar ventana de Hanning antes de FFT |
