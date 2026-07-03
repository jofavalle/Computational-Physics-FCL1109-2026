# Capítulo 6 - Mecánica cuántica
> Referencia: Landau & Páez, *Computational Problems for Physics* (2018), Cap. 6
> Curso: FCL1109 - Física Computacional, UES 2026
> Unidad 5 (semanas 16-17) · Práctica numérica 4

---

## 6.1 Panorama del capítulo

La mecánica cuántica computacional del libro se apoya en tres grandes técnicas numéricas, todas ya vistas en capítulos anteriores:

| Técnica | Sección | Se reutiliza de |
|---------|---------|-----------------|
| EDO (rk4 / Numerov) + búsqueda de raíz | Estados ligados, funciones de onda | Cap. 1 (rk4), Cap. 2 (bisección) |
| Diferencias finitas / leapfrog para la PDE | Schrödinger dependiente del tiempo, paquetes | Cap. 4 (ondas) |
| Álgebra lineal / matrices / Monte Carlo | QM matricial, qubits, integral de camino | Cap. 2, Cap. 7 (Metropolis) |

> 💡 El problema de estados ligados es un **problema de autovalores**: la EDO solo tiene solución física (normalizable) para ciertos valores de la energía. Eso obliga a combinar un integrador de EDO con una búsqueda de raíz sobre `E`.

---

## 6.2 Estados ligados

### 6.2.1 Pozo cuadrado 1-D (semianalítico)

Para el pozo de profundidad `V0` y radio `R` (unidades `ħ = 1, 2m = 1`), las energías de ligadura `E = −E_B < 0` satisfacen las ecuaciones trascendentes:

$$\sqrt{V_0 - E_B}\,\tan\sqrt{V_0 - E_B} = \sqrt{E_B} \quad (\text{par})$$
$$\sqrt{V_0 - E_B}\,\cot\sqrt{V_0 - E_B} = -\sqrt{E_B} \quad (\text{impar})$$

Se resuelven reescribiendo como `f(E_B) = 0` y aplicando **bisección** (la misma de `Bisection.py`, Cap. 2).

```python
import numpy as np

def f_par(EB, V0):
    k = np.sqrt(V0 - EB)
    return k * np.tan(k) - np.sqrt(EB)

def f_impar(EB, V0):
    k = np.sqrt(V0 - EB)
    return k / np.tan(k) + np.sqrt(EB)
```

### 6.2.2 Potencial arbitrario (EDO + búsqueda) - método de *matching*

La ecuación de Schrödinger independiente del tiempo en forma de EDO:

$$\frac{d^2\psi}{dx^2} - \frac{2m}{\hbar^2}V(x)\psi = \kappa^2\psi, \qquad \kappa^2 = \frac{2m}{\hbar^2}|E|$$

Condiciones de frontera de estado ligado (decaimiento exponencial):

$$\psi(x)\to e^{-\kappa x}\ (x\to+\infty), \qquad \psi(x)\to e^{+\kappa x}\ (x\to-\infty)$$

**Algoritmo de *matching*:**
1. Integrar `ψ_L` desde `x = −X` (≈ −∞) hacia la derecha hasta `x_m`.
2. Integrar `ψ_R` desde `x = +X` (≈ +∞) hacia la izquierda hasta `x_m`.
3. Exigir continuidad de la **derivada logarítmica** `ψ'/ψ` en `x_m` (independiente de la normalización):

$$\Delta(E) = \left.\frac{\psi_L'/\psi_L - \psi_R'/\psi_R}{\psi_L'/\psi_L + \psi_R'/\psi_R}\right|_{x=x_m}$$

4. Buscar (bisección) la `E` que hace `Δ(E) = 0`.

> ⚠️ El denominador en `Δ` está solo para acotar la magnitud del desajuste; no tiene significado físico.

### Algoritmo de Numerov (más eficiente que rk4 para EDOs sin primera derivada)

$$\psi(x+h) \simeq \frac{2[1 - \tfrac{5}{12}h^2 k^2(x)]\psi(x) - [1 + \tfrac{1}{12}h^2 k^2(x-h)]\psi(x-h)}{1 + \tfrac{1}{12}h^2 k^2(x+h)}$$

donde `k²(x) = (2m/ħ²)(E − V(x))`. Usa los **dos pasos previos** para avanzar; para integrar hacia atrás se invierte el signo de `h`.

```python
def numerov(k2, u, h):
    b = h**2 / 12.0
    for i in range(1, len(u) - 1):
        u[i+1] = (2*u[i]*(1 - 5*b*k2[i]) - (1 + b*k2[i-1])*u[i-1]) / (1 + b*k2[i+1])
    return u
```

### 6.2.3 Atajo "descuidado" (*sloppy shortcut*)

Integrar hacia afuera desde el origen y verificar que `ψ` sea una exponencial decreciente. **Problema:** la componente `e^{+κx}` introducida por error numérico crece y termina dominando → resultados poco fiables a `x` grande. Útil solo como verificación rápida.

### 6.2.4 Estados ligados relativistas (Klein-Gordon)

Para un pión en un campo coulombiano nuclear `qΦ = −Zα/r`, la KGE radial es:

$$\frac{d^2 u_l}{dr^2} + \left[\frac{2EZ\alpha}{r} - (m^2 - E^2) - \frac{l(l+1) - (Z\alpha)^2}{r^2}\right]u_l = 0$$

- La corrección relativista `l(l+1) → l(l+1) − (Zα)²` rompe la degeneración en `l` (separa 2S y 2P).
- `α = e²/ħc ≈ 1/137` (constante de estructura fina).
- Solución analítica para comparar:

$$E_{n,l} \simeq m - \frac{m(Z\alpha)^2}{2n^2} - \frac{m(Z\alpha)^4}{2n^4}\left(\frac{n}{l+\tfrac12} - \frac34\right) + O(\alpha^6)$$

> ⚠️ El radical de la fórmula exacta suma números muy grandes y muy pequeños → evaluar mejor la serie término a término.

---

## 6.3 Simulación de decaimiento espontáneo (Monte Carlo)

Ley básica (en diferencias finitas, no la EDO):

$$P = \frac{\Delta N(t)/N(t)}{\Delta t} = -\lambda \quad\Rightarrow\quad N(t) \simeq N(0)e^{-\lambda t}$$

**Algoritmo:** en cada paso de tiempo, para cada átomo vivo se compara un número aleatorio con `λ`; si `r < λ` el átomo decae.

```python
import numpy as np

def decaimiento(N0, lam, t_max):
    N = N0
    historia = []
    for t in range(t_max + 1):
        dN = np.sum(np.random.random(N) < lam)   # cuántos decaen este paso
        N -= dN
        historia.append((t, N, dN))
        if N == 0:
            break
    return historia
```

- Graficar `ln N(t)` y `ln(ΔN/Δt)` vs `t` → pendiente proporcional a `λ`.
- Con `N(0)` grande parece exponencial puro; al bajar `N(t)` aparece la naturaleza estocástica (más fluctuación en `ΔN`).

### 6.3.1 Ajuste del espectro de cuerpo negro (datos COBE)

$$I(\nu, T) = \frac{2h\nu^3}{c^2}\frac{1}{e^{h\nu/kT} - 1}$$

Aplicar logaritmo permite un ajuste lineal por mínimos cuadrados (Cap. 2) para deducir `T` del fondo cósmico de microondas (≈ 2.7 K).

---

## 6.4 Funciones de onda

### 6.4.1 Oscilador armónico

Forma adimensional (energía ya incorporada vía `n`):

$$\frac{d^2u}{dx^2} + (2n + 1 - x^2)u = 0, \qquad n = 0, 1, 2, \dots$$

Como la energía ya está en la ecuación, **no hace falta buscar**: se integra directo con rk4. Condiciones iniciales según paridad:

| Paridad | `y[0] = ψ(0)` | `y[1] = ψ'(0)` |
|---------|---------------|----------------|
| Par     | 1             | 0              |
| Impar   | 0             | 1              |

```python
# RHS para rk4: y[0]=psi, y[1]=psi'
def f_HO(x, y, n):
    return np.array([y[1], -(2*n + 1 - x**2) * y[0]])
```

- Para `n` grande (20-30) la densidad `|ψ|²` se acumula cerca de los **puntos de retorno clásicos**.
- Verificación analítica: `ψ_n(x) = H_n(x) e^{−x²/2}` (polinomios de Hermite).

---

## 6.5 Expansión en ondas parciales

$$e^{i\mathbf{k}\cdot\mathbf{r}} = \sum_{\ell=0}^{\infty}(2\ell+1)i^\ell j_\ell(kr)P_\ell(\cos\theta)$$

### 6.5.1 Polinomios asociados de Legendre

$$\frac{d}{dx}\left[(1-x^2)\frac{dP_\ell^m}{dx}\right] + \left[\ell(\ell+1) - \frac{m^2}{1-x^2}\right]P_\ell^m(x) = 0$$

Se resuelve como EDO con rk4 (con `x = cosθ`). Forma dinámica:

```python
def f_legendre(c, y, el, m):
    # c = cos(theta); y[0]=Plm, y[1]=dPlm/dc
    rhs1 = (2*c*y[1]/(1 - c**2)
            - (el*(el+1) - m**2/(1 - c**2)) * y[0] / (1 - c**2))
    return np.array([y[1], rhs1])
```

Armónicos esféricos a partir de `P_ℓ^m`:

$$Y_\ell^m(\theta,\phi) = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}\,P_\ell^m(\cos\theta)\,e^{im\phi}$$

Verificar la ortonormalidad `∫∫ Y_ℓ^m* Y_ℓ'^m' = δ_ℓℓ' δ_mm'` (doble integral anidada).

---

## 6.6 Funciones de onda del hidrógeno

$$\psi_{n\ell m}(r,\theta,\phi) = R_{n\ell}(r)\,Y_\ell^m(\theta,\phi)$$

Con `R_{nℓ}(ρ) = F_{nℓ}(ρ)e^{−ρ/2}`, la EDO para `F` es:

$$\frac{d^2F_{n\ell}}{d\rho^2} + \left(\frac{2}{\rho} - 1\right)\frac{dF_{n\ell}}{d\rho} + \left[\frac{\lambda-1}{\rho} - \frac{\ell(\ell+1)}{\rho^2}\right]F_{n\ell} = 0$$

con `λ = n = 1,2,3,…` y `ℓ < n`. Densidad radial:

$$P(r) = 4\pi r^2 |F_{n\ell}(\rho)e^{-\rho/2}|^2, \qquad \int_0^\infty P(r)\,dr = 1$$

```python
def f_hidrogeno(r, y, n, el):
    # y[0]=F, y[1]=F'
    rhs1 = -(2/r - 1)*y[1] - ((n - 1)/r - el*(el+1)/r**2) * y[0]
    return np.array([y[1], rhs1])
```

---

## 6.7 Paquetes de onda y Schrödinger dependiente del tiempo

### 6.7.1 Paquete en el oscilador armónico

Paquete cuyo centro oscila con el periodo clásico (no cambia de forma):

$$|\psi(x,t)|^2 = \frac{\alpha}{\sqrt{\pi}}\,e^{-\alpha^2[x - a\cos(\omega t)]^2}$$

### 6.7.3 Solución directa de la PDE (leapfrog) - **núcleo del capítulo**

Se escribe `ψ = R + iI` y la ecuación se desdobla en dos PDEs acopladas. Algoritmo de diferencias finitas (futuro = presente + cambio), con `β = Δt/Δx²`:

$$R_i^{n+1} = R_i^n - \beta(I_{i+1}^n + I_{i-1}^n - 2I_i^n) + \Delta t\,V_i\,I_i^n$$
$$I_i^{n+1} = I_i^n + \beta(R_{i+1}^n + R_{i-1}^n - 2R_i^n) - \Delta t\,V_i\,R_i^n$$

> 💡 Se resuelven R e I en tiempos **escalonados** (medio paso de diferencia) para conservar mejor la probabilidad. Solo hay que guardar presente y futuro, no todos los tiempos.

```python
import numpy as np

def schrodinger_1d(R, I, V, dx, dt, n_pasos):
    beta = dt / dx**2
    for _ in range(n_pasos):
        R[1:-1] -= beta*(I[2:] + I[:-2] - 2*I[1:-1]) - dt*V[1:-1]*I[1:-1]
        I[1:-1] += beta*(R[2:] + R[:-2] - 2*R[1:-1]) - dt*V[1:-1]*R[1:-1]
    return R, I

# Paquete inicial típico: gaussiana por onda plana
# psi(x,0) = exp(-0.5*((x-x0)/sigma)**2) * exp(i*k0*x)
```

Condiciones de frontera: `ψ = 0` en los bordes de una caja grande. Verificar conservación de `∫|ψ|²dx`. Hacer ≥ 5000 pasos en producción.

### 6.7.4 Con campo eléctrico externo

Se añade un término al potencial: `V = ½kx² − E·x` (carga `q=1`). Variar la fuerza del oscilador y la frecuencia del campo sinusoidal para ver **resonancia**.

---

## 6.8 Dispersión (scattering)

> En dispersión la energía es **positiva y continua** (no hay autovalores); las funciones de onda no son normalizables.

### 6.8.1 Pozo cuadrado 3-D

Se resuelve por ondas parciales. Función externa = combinación de Bessel `j_ℓ` y Neumann `n_ℓ` esféricas con un **corrimiento de fase** `δ_ℓ`:

$$\tan\delta_\ell = \frac{k j_\ell'(ka) - \gamma_\ell j_\ell(ka)}{k n_\ell'(ka) - \gamma_\ell n_\ell(ka)}, \qquad \gamma_\ell = \kappa\frac{j_\ell'(\kappa a)}{j_\ell(\kappa a)}$$

Secciones eficaces:

$$\frac{d\sigma}{d\Omega}(\theta) = \frac{1}{k^2}\left|\sum_\ell (2\ell+1)e^{i\delta_\ell}\sin\delta_\ell\,P_\ell(\cos\theta)\right|^2, \qquad \sigma_{tot} = \frac{4\pi}{k^2}\sum_\ell(2\ell+1)\sin^2\delta_\ell$$

> 💡 Usar `scipy.special.spherical_jn` / `spherical_yn` para Bessel y Neumann. **Efecto Ramsauer-Townsend:** a cierta energía baja `σ_tot` casi se anula (ej. `V0=30.1`, `E≈5.36`).

### 6.8.2 Dispersión de Coulomb

El potencial `1/r` no se anula a `r→∞` → requiere la **función hipergeométrica confluente** `₁F₁`:

$$f_\ell(r) = C_\ell\,{}_1F_1(\ell + 1 + i\eta,\ 2\ell + 2,\ -2ikr), \qquad \eta = \mu ZZ'e^2/\hbar^2 k$$

> 💡 `scipy` **no** tiene `₁F₁` para argumento complejo; usar `mpmath.hyp1f1`. La función gamma compleja sí está en `scipy.special.gamma`.

### 6.8.3 / 6.8.4 Tres discos y billares cuánticos (caos cuántico)

Se resuelve la PDE dependiente del tiempo en 2-D (leapfrog) con un paquete gaussiano incidente:

$$R_{i,j}^{n+1} = R_{i,j}^n - \frac{\Delta t}{\Delta x^2}\left(I_{i+1,j}^n + I_{i-1,j}^n - 4I_{i,j}^n + I_{i,j+1}^n + I_{i,j-1}^n\right) - \Delta t\,V_{i,j}I_{i,j}^n$$

- Discos duros: imponer `ψ = 0` dentro de cada disco.
- Para *scattering* agrandar la caja (evitar reflexiones del borde); para billar las reflexiones del borde son el fenómeno buscado.
- Alto grado de dispersión múltiple (señal de caos) cuando `a/R ≈ 6`.

---

## 6.9 Mecánica cuántica matricial

### 6.9.1-6.9.3 Estados ligados en espacio de momentos (ecuación integral)

La ecuación de Schrödinger en espacio `k` es una **ecuación integral** que se discretiza con cuadratura de Gauss en `N` puntos `k_j` con pesos `w_j`, convirtiéndola en un problema matricial de autovalores `[H][ψ] = E[ψ]`:

$$H_{ij} = \frac{k_i^2}{2\mu}\delta_{ij} + \frac{2}{\pi}w_j k_j^2 V(k_i, k_j)$$

**Potencial delta-shell** `V(r) = (λ/2μ)δ(r−b)` (uno de los pocos con solución analítica):

$$V(k', k) = \frac{\lambda}{2\mu}\frac{\sin(k'b)\sin(kb)}{k'k}, \qquad e^{-2\kappa b} - 1 = \frac{2\kappa}{\lambda}$$

```python
import numpy as np
# Requiere puntos/pesos de Gauss (GaussPoints en el libro; usar np.polynomial.legendre.leggauss)
def hamiltoniano_kspace(k, w, V, mu):
    N = len(k)
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H[i, j] = (2/np.pi) * w[j] * k[j]**2 * V(k[i], k[j])
            if i == j:
                H[i, j] += k[i]**2 / (2*mu)
    E, vecs = np.linalg.eig(H)
    return E, vecs
```

> ⚠️ El solver devuelve varios autovalores; los estados ligados reales están en energías **negativas** y cambian poco al variar `N`. Los demás son artefactos numéricos. Subir `N` de 8 en 8 (16, 24, 32…) y observar convergencia antes de que el redondeo introduzca fluctuaciones.

### 6.9.4 Estructura hiperfina del hidrógeno (Sympy)

Interacción espín electrón-protón `V = W σ_e · σ_p` con matrices de Pauli. El hamiltoniano `4×4` da autovalores `−3W` (triplete, mult. 3) y `W` (singlete). El desdoblamiento del estado 1S predice la línea de **1420 MHz** (21 cm). Con campo `B` externo se añade `H' = (μ_e σ_z^e + μ_p σ_z^p)B`. Se resuelve simbólicamente con **Sympy** (`Matrix(...).eigenvals()`).

### 6.9.5 Simetría SU(3) de los quarks

Los 8 generadores de Gell-Mann `λ_i` (extensión de Pauli a 3-D) actúan sobre la base de quarks `|u⟩, |d⟩, |s⟩`. Operadores de subida/bajada:

$$I_\pm = \tfrac12(\lambda_1 \pm i\lambda_2),\quad U_\pm = \tfrac12(\lambda_6 \pm i\lambda_7),\quad V_\pm = \tfrac12(\lambda_4 \pm i\lambda_5)$$

Verificar con numpy/scipy: unitariedad (`U†U=1`), `det=1`, relaciones de conmutación `[T_a,T_b]=if_{abc}T_c`, y el Casimir `Σλ_iλ_i = 16/3`.

---

## 6.10 Estados coherentes y entrelazamiento

### 6.10.1 Estados coherentes de Glauber

Superposición de estados del oscilador que es autoestado del operador de aniquilación:

$$|\alpha\rangle = e^{-\alpha^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle, \qquad a|\alpha\rangle = \alpha|\alpha\rangle,\quad E_\alpha = \alpha^2 + \tfrac12$$

Se construye sumando hasta `n_max` con `⟨x|n⟩ = H_n(βx)e^{−β²x²/2}` (Hermite). Se comporta como un paquete que oscila sin cambiar de forma.

### 6.10.2 Kaones neutros como superposición de estados

`K⁰` y `K̄⁰` se crean por la interacción fuerte pero decaen por la débil como autoestados de CP `K_1` (→2π) y `K_2` (→3π) con vidas muy distintas:

$$|K^0(t)\rangle = |K_1\rangle e^{-t/2\tau_1} + |K_2\rangle e^{-t/2\tau_2}$$

La pequeña violación de CP (`ε ≈ 0.0023`) introduce además oscilación por la diferencia de masas `Δm`. La probabilidad de decaimiento a 2π no es exponencial pura (combina exponenciales + término periódico).

### 6.10.3 Transiciones en doble pozo

Una perturbación `ΔE` que baja la barrera permite la transición `P(L→R) = sin²(ΔE·t/ħ)` - la esencia de las oscilaciones de kaones. Se simula con la PDE dependiente del tiempo (`TwoWells.py`).

### 6.10.4 Qubits y entrelazamiento

Un qubit es un sistema de dos estados `|0⟩, |1⟩`. Estados de dos qubits:
- **Separable:** se factoriza como producto tensorial, p. ej. `(|01⟩+|10⟩)/√2`.
- **Entrelazado:** no se factoriza, p. ej. `(|00⟩+|11⟩)/√2`.

Criterio en `C⁴` para `(w,x,y,z)`: separable ⟺ `wz = xy`.

Interacción dipolo-dipolo en el espacio producto:

$$H = \frac{\mu^2}{r^3}(X_A\otimes X_B + Y_A\otimes Y_B - 2Z_A\otimes Z_B) = \frac{\mu^2}{r^3}\begin{pmatrix}-2&0&0&0\\0&2&2&0\\0&2&2&0\\0&0&0&-2\end{pmatrix}$$

Autovalores `4, 0, −2, −2`; los autovectores `(|01⟩±|10⟩)/√2` son **entrelazados**, `|00⟩` y `|11⟩` son **separables**. Se resuelve con `numpy.linalg.eig` y producto tensorial `np.kron`.

```python
import numpy as np
X = np.array([[0,1],[1,0]]);  Y = np.array([[0,-1j],[1j,0]]);  Z = np.array([[1,0],[0,-1]])
H = np.kron(X,X) + np.kron(Y,Y).real - 2*np.kron(Z,Z)   # factor mu^2/r^3 aparte
E, vecs = np.linalg.eig(H)
```

---

## 6.11 Integral de camino de Feynman (Monte Carlo cuántico)

La amplitud de propagación se expresa como suma sobre **todos los caminos** ponderados por la acción:

$$G(b,a) = \sum_{\text{caminos}} e^{iS[b,a]/\hbar}, \qquad S = \int_{t_a}^{t_b} L\,dt, \quad L = T - V$$

En tiempo imaginario, la densidad del estado fundamental se obtiene minimizando la energía discretizada de la trayectoria con el **algoritmo de Metropolis** (Cap. 7):

$$|\psi_0(x)|^2 = \frac{1}{Z}\lim_{\tau\to\infty}\int dx_1\cdots dx_{N-1}\,e^{-\varepsilon E}$$

```python
# Energía discreta de un eslabón: E = KE + PE = (Δx)^2 + x^2
def energia_camino(path):
    dx2 = np.sum((path[1:] - path[:-1])**2)
    return dx2 + path[-1]**2

# Metropolis: cambiar un elemento al azar; aceptar si baja E,
# o si exp(-(newE-oldE)) > random  ->  acumular en histograma de probabilidad
```

El camino clásico (mínima acción) es el más probable; los caminos cercanos contribuyen menos.

---

## Métodos numéricos del capítulo (resumen)

| Método | Dónde se usa | Notas |
|--------|--------------|-------|
| rk4 + bisección (matching) | Estados ligados, potencial arbitrario | Autovalores por derivada logarítmica |
| Numerov | EDO sin 1ª derivada (Schrödinger) | Más preciso que rk4 a igual `h` |
| Monte Carlo (aleatorio) | Decaimiento espontáneo | Ley en diferencias finitas |
| Leapfrog / diferencias finitas | Schrödinger dependiente del tiempo | R e I en tiempos escalonados |
| Cuadratura de Gauss + autovalores | QM en espacio de momentos | Ecuación integral → matriz |
| Álgebra lineal (eig) / Sympy | Hiperfina, SU(3), qubits | Matrices de Pauli, producto tensorial |
| Metropolis | Integral de camino de Feynman | Estado fundamental por tiempo imaginario |

---

## Constantes y unidades útiles

```python
# El libro usa frecuentemente unidades naturales:
#   hbar = 1,  c = 1  (relativista)  o  hbar = 1, 2m = 1  (pozo 1-D)
hbarc   = 197.33        # MeV·fm  (para problemas nucleares/KGE)
alpha   = 1/137.0       # constante de estructura fina
# Factor recurrente en los listings nucleares:  2m c^2 / (hbar c)^2
fact    = 2*940 / 197.33**2   # ≈ 0.04829  (m ≈ 940 MeV, nucleón)
```

---

## Problemas típicos del libro (Cap. 6)

- **6.2** Estados ligados del pozo: bisección (semianalítico) vs rk4-matching vs Numerov
- **6.2.4** Klein-Gordon: estados del bario piónico (`Z=56`), separación 2S-2P
- **6.3** Decaimiento espontáneo Monte Carlo: `ln N(t)` vs `t`, fluctuaciones
- **6.4** Oscilador armónico: paridad, puntos de retorno clásicos, comparar con Hermite
- **6.5/6.6** Legendre `P_ℓ^m`, armónicos esféricos, densidad radial del hidrógeno
- **6.7** Paquete de onda en pozo/oscilador: PDE leapfrog, conservación de probabilidad
- **6.8** Dispersión pozo cuadrado: corrimientos de fase, Ramsauer-Townsend
- **6.9** Estado ligado delta-shell en espacio `k`; estructura hiperfina (1420 MHz)
- **6.10** Estado de Glauber; kaones; doble pozo; qubits entrelazados vs separables
- **6.11** Integral de camino de Feynman con Metropolis

---

## Notas para adaptar los listings del libro

- Los listings que usan `rk4Algor(x, h, n, y, f)` deben adaptarse a `fcl1109.rk4(f, x, y, h)`.
- `GaussPoints` del libro → `np.polynomial.legendre.leggauss` (reescalar al intervalo `[min, max]`).
- Los listings con `from visual import *` / `vpython` usan animación VPython; preferir las versiones `*Mat.py` con Matplotlib.
- `scipy.special.sph_jn/sph_yn` están **obsoletos**: usar `scipy.special.spherical_jn(n, x)` y `spherical_yn(n, x, derivative=...)`.
