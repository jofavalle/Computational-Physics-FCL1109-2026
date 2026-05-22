# Capítulo 3 — Dinámica clásica y no lineal
> Referencia: Landau & Páez, *Computational Problems for Physics* (2018), Cap. 3  
> Curso: FCO4101 — Física Computacional, UES 2026

---

## 3.1 Oscilador armónico y anarmónico

### Ecuaciones de movimiento
El oscilador armónico simple satisface:

$$\ddot{x} = -\omega_0^2 x$$

Forma de sistema de primer orden (necesaria para RK4):

$$\frac{dx}{dt} = v, \qquad \frac{dv}{dt} = -\omega_0^2 x$$

El oscilador anarmónico (con no linealidad cúbica):

$$\frac{dv}{dt} = -\omega_0^2 x - \epsilon x^3$$

### Implementación típica en Python
```python
def derivadas_oscilador(t, estado, omega0, epsilon=0.0):
    x, v = estado
    dxdt = v
    dvdt = -omega0**2 * x - epsilon * x**3
    return [dxdt, dvdt]
```

---

## 3.2 Péndulo simple (no lineal)

### Ecuación exacta (sin aproximación de ángulo pequeño)

$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta$$

Sistema de primer orden:

$$\frac{d\theta}{dt} = \omega, \qquad \frac{d\omega}{dt} = -\frac{g}{L}\sin\theta$$

> ⚠️ Landau insiste en usar `sin(θ)` exacto, no la aproximación lineal `θ`. La diferencia se vuelve significativa para amplitudes > 15°.

### Energía del péndulo (verificación de conservación)
$$E = \frac{1}{2}mL^2\omega^2 + mgL(1 - \cos\theta)$$

```python
def energia_pendulo(theta, omega, L=1.0, m=1.0, g=9.8):
    T = 0.5 * m * L**2 * omega**2
    V = m * g * L * (1 - np.cos(theta))
    return T + V
```

---

## 3.3 Péndulo forzado y amortiguado

### Ecuación general
$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta - q\frac{d\theta}{dt} + F_d\cos(\Omega_d t)$$

Parámetros clave:
| Símbolo | Significado | Valor típico (Landau) |
|---------|-------------|----------------------|
| `q` | coeficiente de amortiguamiento | 0.5 |
| `F_d` | amplitud de fuerza externa | 0.5 – 1.2 |
| `Omega_d` | frecuencia de forzamiento | 2/3 |

```python
def derivadas_pendulo_forzado(t, estado, q, F_d, Omega_d, g=9.8, L=1.0):
    theta, omega = estado
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - q*omega + F_d*np.cos(Omega_d*t)
    # Mantener theta en [-pi, pi]
    return [dtheta, domega]
```

> 💡 Landau recomienda siempre mantener θ en el rango [−π, π] sumando/restando 2π cuando sea necesario.

---

## 3.4 Caos y diagramas de Poincaré

### Cuándo aparece el caos
El péndulo forzado entra en régimen caótico típicamente cuando `F_d > 1.07` (con q=0.5, Ωd=2/3). El exponente de Lyapunov positivo confirma caos.

### Diagrama de Poincaré
Se muestrea el estado `(θ, ω)` una vez por periodo de forzamiento T = 2π/Ωd:

```python
T_forzamiento = 2 * np.pi / Omega_d
tiempos_poincare = np.arange(0, t_max, T_forzamiento)

puntos_poincare = []
for t_p in tiempos_poincare:
    idx = np.argmin(np.abs(t_array - t_p))
    theta_p = theta_array[idx] % (2*np.pi)  # normalizar
    puntos_poincare.append((theta_p, omega_array[idx]))
```

### Exponente de Lyapunov (estimación numérica)
```python
# Integrar dos trayectorias casi idénticas
# delta_0: separación inicial pequeña (~1e-8)
# lambda = (1/t) * ln(|delta(t)| / delta_0)
lambda_lyapunov = (1/t_final) * np.log(separacion_final / delta_0)
```

---

## 3.5 Mapa logístico (dinámica discreta)

$$x_{n+1} = r \, x_n (1 - x_n)$$

- `r < 3.0`: punto fijo estable
- `3.0 < r < 3.57`: bifurcaciones periódicas
- `r > 3.57`: caos

```python
def mapa_logistico(r, x0, n_iter):
    x = x0
    valores = []
    for _ in range(n_iter):
        x = r * x * (1 - x)
        valores.append(x)
    return np.array(valores)
```

**Diagrama de bifurcación:**
```python
r_vals = np.linspace(2.5, 4.0, 1000)
for r in r_vals:
    x = 0.5
    for _ in range(200):   # transiente
        x = r * x * (1 - x)
    for _ in range(100):   # régimen estacionario
        x = r * x * (1 - x)
        plt.plot(r, x, ',k', alpha=0.3)
```

---

## 3.6 Integración numérica — RK4 (método preferido del curso)

### Fórmula general

$$k_1 = h \, f(t_n,\, y_n)$$
$$k_2 = h \, f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{k_1}{2}\right)$$
$$k_3 = h \, f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{k_2}{2}\right)$$
$$k_4 = h \, f(t_n + h,\, y_n + k_3)$$
$$y_{n+1} = y_n + \tfrac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

```python
def rk4_paso(f, t, y, h, **params):
    k1 = h * np.array(f(t,       y,           **params))
    k2 = h * np.array(f(t + h/2, y + k1/2,   **params))
    k3 = h * np.array(f(t + h/2, y + k2/2,   **params))
    k4 = h * np.array(f(t + h,   y + k3,     **params))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def integrar_rk4(f, y0, t0, t_max, h, **params):
    t_vals = [t0]
    y_vals = [np.array(y0)]
    t, y = t0, np.array(y0)
    while t < t_max:
        y = rk4_paso(f, t, y, h, **params)
        t += h
        t_vals.append(t)
        y_vals.append(y.copy())
    return np.array(t_vals), np.array(y_vals)
```

> 💡 También se puede usar `scipy.integrate.solve_ivp(f, [t0, tf], y0, method='RK45')` para problemas donde el paso adaptativo es importante.

---

## 3.7 Verificaciones y buenas prácticas (según Landau)

| Verificación | Cómo hacerla |
|---|---|
| Conservación de energía | Graficar E(t); debe ser constante |
| Convergencia del paso h | Comparar soluciones con h y h/2 |
| Condiciones iniciales | Probar varios y0 para explorar el espacio de fase |
| Normalización de θ | Siempre mapear a [−π, π] en péndulos |

---

## Problemas típicos del libro (Cap. 3)

- **3.1** Oscilador anarmónico: comparar periodo numérico vs analítico
- **3.2** Péndulo: graficar espacio de fase para varias amplitudes
- **3.3** Péndulo forzado: trazar diagrama de Poincaré en régimen caótico
- **3.4** Mapa logístico: diagrama de bifurcación completo
- **3.5** Calcular exponente de Lyapunov para F_d = 1.2
