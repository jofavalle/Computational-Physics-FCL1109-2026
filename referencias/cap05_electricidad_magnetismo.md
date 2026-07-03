# Capítulo 5 - Electricidad y magnetismo
> Referencia: Landau & Páez, *Computational Problems for Physics* (2018), Cap. 5  
> Curso: FCL1109 - Física Computacional, UES 2026

---

## 5.1 Potencial electrostático - Ecuación de Laplace y Poisson

### Ecuación de Laplace (región sin cargas)
$$\nabla^2 V = 0$$

### Ecuación de Poisson (con distribución de carga)
$$\nabla^2 V = -\frac{\rho(\mathbf{r})}{\epsilon_0}$$

### Relación campo-potencial
$$\mathbf{E} = -\nabla V$$

En 2D discreto:
```python
def campo_electrico_2d(V, dx, dy):
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)
    # Diferencias centradas (nodos interiores)
    Ex[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2*dx)
    Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2*dy)
    return Ex, Ey
```

---

## 5.2 Métodos de relajación para Laplace 2D

### Jacobi
```python
def jacobi(V, rho, dx, epsilon_0=8.854e-12, tol=1e-5):
    while True:
        V_nuevo = V.copy()
        V_nuevo[1:-1, 1:-1] = 0.25 * (
            V[2:, 1:-1] + V[:-2, 1:-1] +
            V[1:-1, 2:] + V[1:-1, :-2]
            + dx**2 * rho[1:-1, 1:-1] / epsilon_0
        )
        if np.max(np.abs(V_nuevo - V)) < tol:
            break
        V = V_nuevo
    return V
```

### Gauss-Seidel (converge ~2× más rápido que Jacobi)
```python
def gauss_seidel(V, rho, dx, epsilon_0=8.854e-12, tol=1e-5):
    while True:
        V_anterior = V.copy()
        for i in range(1, V.shape[0]-1):
            for j in range(1, V.shape[1]-1):
                V[i,j] = 0.25 * (V[i+1,j] + V[i-1,j] +
                                   V[i,j+1] + V[i,j-1]
                                   + dx**2 * rho[i,j] / epsilon_0)
        if np.max(np.abs(V - V_anterior)) < tol:
            break
    return V
```

### SOR - Successive Over-Relaxation (más rápido)
$$V_{i,j}^{\text{nuevo}} = (1-\omega)V_{i,j}^{\text{old}} + \frac{\omega}{4}\left(V_{i+1,j}+V_{i-1,j}+V_{i,j+1}+V_{i,j-1}\right)$$

```python
omega = 1.9  # parámetro óptimo típico: 1.5 < omega < 2.0

def sor(V, rho, dx, omega=1.9, tol=1e-5):
    while True:
        V_anterior = V.copy()
        for i in range(1, V.shape[0]-1):
            for j in range(1, V.shape[1]-1):
                V_gs = 0.25 * (V[i+1,j] + V[i-1,j] +
                                V[i,j+1] + V[i,j-1])
                V[i,j] = (1-omega)*V[i,j] + omega*V_gs
        if np.max(np.abs(V - V_anterior)) < tol:
            break
    return V
```

---

## 5.3 Condiciones de frontera (EM)

| Geometría | Condición típica |
|-----------|-----------------|
| Conductor plano a potencial V₀ | Dirichlet: `V[borde] = V0` |
| Frontera abierta / al infinito | Neumann: `dV/dn = 0` → `V[0,:] = V[1,:]` |
| Caja conductora con tapa caliente | Tres lados a 0V, uno a V₀ |
| Condensador de placas paralelas | Placa inferior V=0, placa superior V=V₀ |

```python
# Ejemplo: caja con pared superior a V0
def aplicar_frontera_caja(V, V0=1.0):
    V[0, :]  = 0.0   # borde inferior
    V[-1, :] = V0    # borde superior
    V[:, 0]  = 0.0   # borde izquierdo
    V[:, -1] = 0.0   # borde derecho
    return V
```

---

## 5.4 Ley de Biot-Savart (campo magnético)

### Elemento diferencial de corriente

$$d\mathbf{B} = \frac{\mu_0}{4\pi} \frac{I\, d\mathbf{l} \times \hat{r}}{r^2}$$

### Campo en el eje de una espira circular

$$B_z = \frac{\mu_0 I R^2}{2(R^2 + z^2)^{3/2}}$$

```python
def campo_espira(z_array, R, I, mu0=4*np.pi*1e-7):
    Bz = mu0 * I * R**2 / (2 * (R**2 + z_array**2)**1.5)
    return Bz
```

### Campo de par de bobinas de Helmholtz

Dos espiras separadas una distancia `d = R` (configuración óptima para campo uniforme):

```python
def helmholtz(z_array, R, I, mu0=4*np.pi*1e-7):
    d = R / 2  # cada bobina a ±d del centro
    B1 = mu0 * I * R**2 / (2 * (R**2 + (z_array - d)**2)**1.5)
    B2 = mu0 * I * R**2 / (2 * (R**2 + (z_array + d)**2)**1.5)
    return B1 + B2
```

---

## 5.5 Ondas electromagnéticas - Ecuaciones de Maxwell

### Forma 1D (polarización en x, propagación en z)

$$\frac{\partial E_x}{\partial t} = -\frac{1}{\mu_0}\frac{\partial B_y}{\partial z}, \qquad \frac{\partial B_y}{\partial t} = -\frac{1}{\epsilon_0}\frac{\partial E_x}{\partial z} \cdot \frac{1}{c^2}$$

### Algoritmo FDTD (Finite-Difference Time-Domain) - Yee scheme

E y B se evalúan en puntos de grilla intercalados (media celda de diferencia):

```python
def fdtd_1d(Ez, Hy, eps0, mu0, dx, dt, n_pasos):
    c = 1.0 / np.sqrt(eps0 * mu0)
    courant = c * dt / dx  # debe ser <= 1
    
    for _ in range(n_pasos):
        # Actualizar H (medio paso adelante)
        Hy[:-1] -= (dt / (mu0 * dx)) * (Ez[1:] - Ez[:-1])
        # Actualizar E (paso completo)
        Ez[1:-1] -= (dt / (eps0 * dx)) * (Hy[1:] - Hy[:-1])
    return Ez, Hy
```

---

## 5.6 Capacitancia numérica

La capacitancia se puede calcular numéricamente una vez que se conoce V:

$$Q = \epsilon_0 \oint \mathbf{E} \cdot d\mathbf{A} \approx \epsilon_0 \sum_{\text{sup}} E_n \Delta A$$

$$C = \frac{Q}{V}$$

```python
def capacitancia_numerica(V, dx, dy, epsilon_0=8.854e-12, V_placa=1.0):
    # Calcular campo eléctrico normal en la superficie de la placa superior
    Ey = -(V[-1, :] - V[-2, :]) / dy
    Q = epsilon_0 * np.sum(Ey) * dx
    return Q / V_placa
```

---

## 5.7 Visualización recomendada para EM

```python
import matplotlib.pyplot as plt
import numpy as np

def graficar_potencial_y_campo(x, y, V, Ex, Ey):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa de color del potencial
    im = axes[0].contourf(x, y, V.T, levels=50, cmap='RdBu_r')
    axes[0].contour(x, y, V.T, levels=20, colors='k', linewidths=0.5)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title('Potencial V (Volt)')
    
    # Líneas de campo eléctrico
    axes[1].streamplot(x, y, Ex.T, Ey.T, density=1.5,
                       color=np.sqrt(Ex.T**2 + Ey.T**2),
                       cmap='plasma')
    axes[1].set_title('Líneas de campo E')
    
    plt.tight_layout()
    return fig
```

---

## 5.8 Constantes físicas útiles

```python
# Constantes SI
epsilon_0 = 8.854187817e-12   # F/m - permitividad del vacío
mu_0      = 4 * np.pi * 1e-7  # H/m - permeabilidad del vacío
c_luz     = 2.99792458e8      # m/s - velocidad de la luz
k_e       = 1 / (4*np.pi*epsilon_0)  # N·m²/C² - constante de Coulomb
```

---

## Problemas típicos del libro (Cap. 5)

- **5.1** Potencial en caja conductora 2D: una pared a V₀, resto a 0. Comparar Jacobi vs SOR
- **5.2** Condensador de placas paralelas: calcular capacitancia y comparar con C = ε₀A/d
- **5.3** Campo de espira y bobinas de Helmholtz: trazar uniformidad en el eje
- **5.4** Carga puntual en una caja: resolver Poisson y trazar equipotenciales
- **5.5** FDTD 1D: propagar pulso gaussiano, observar reflexión en conductor perfecto

---

## Comparación de métodos de relajación

| Método | Convergencia | Implementación | Uso recomendado |
|--------|-------------|----------------|-----------------|
| Jacobi | Lenta O(N²) | Más simple | Solo para aprender |
| Gauss-Seidel | ~2× Jacobi | Moderada | Grillas medianas |
| SOR (ω≈1.9) | ~10× Jacobi | Moderada | Producción |
| scipy.sparse | Muy rápida | Compleja | Grillas grandes |

```python
# Alternativa con scipy para grillas grandes
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Construir matriz del sistema y resolver directamente
# (ver Landau Cap. 5 para la construcción completa)
```
