''' Script para testear el funcionamimiento de la librería fcl1109.py '''
import fcl1109 as fc
import numpy as np

# --- Funciones de prueba ---

def f(x):
    return x**3  # f'(x) = 3x², f''(x) = 6x, f'''(x) = 6

def g(x):
    return np.sin(x)  # g'(x) = cos(x)

x0 = 2.0
h = 1e-5

# Test derivada_adelante (derivada hacia delante)
dd = fc.derivada_adelante(f, x0, h)
esperado_d = 3 * x0**2  # f'(2) = 12
print(f"derivada_adelante(x³, x={x0}, h={h}) = {dd:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(dd - esperado_d):.2e}")
assert abs(dd - esperado_d) < 1e-4, "derivada_adelante falló para x³"
print("  ✓ derivada_adelante OK\n")

# Test derivada_central (derivada central)
dc = fc.derivada_central(f, x0, h)
print(f"derivada_central(x³, x={x0}, h={h}) = {dc:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(dc - esperado_d):.2e}")
assert abs(dc - esperado_d) < 1e-8, "derivada_central falló para x³"
print("  ✓ derivada_central OK\n")

# Test derivada_central con sin(x)
dc_sin = fc.derivada_central(g, np.pi/4, h)
esperado_cos = np.cos(np.pi/4)
print(f"derivada_central(sin, x=π/4, h={h}) = {dc_sin:.10f}")
print(f"  Valor esperado: {esperado_cos:.10f}")
print(f"  Error: {abs(dc_sin - esperado_cos):.2e}")
assert abs(dc_sin - esperado_cos) < 1e-8, "derivada_central falló para sin(x)"
print("  ✓ derivada_central con sin(x) OK\n")

# Test derivada_enesima — orden 0 (debe devolver f(x))
nd0 = fc.derivada_enesima(f, x0, 0)
print(f"derivada_enesima(x³, x={x0}, n=0) = {nd0}")
print(f"  Valor esperado: {f(x0)}")
assert nd0 == f(x0), "derivada_enesima n=0 falló"
print("  ✓ derivada_enesima n=0 OK\n")

# Test derivada_enesima — primera derivada
nd1 = fc.derivada_enesima(f, x0, 1)
print(f"derivada_enesima(x³, x={x0}, n=1) = {nd1:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(nd1 - esperado_d):.2e}")
assert abs(nd1 - esperado_d) < 1e-4, "derivada_enesima n=1 falló"
print("  ✓ derivada_enesima n=1 OK\n")

# Test derivada_enesima — segunda derivada
nd2 = fc.derivada_enesima(f, x0, 2)
esperado_2 = 6 * x0  # f''(2) = 12
print(f"derivada_enesima(x³, x={x0}, n=2) = {nd2:.6f}")
print(f"  Valor esperado: {esperado_2}")
print(f"  Error: {abs(nd2 - esperado_2):.2e}")
assert abs(nd2 - esperado_2) < 1e-2, "derivada_enesima n=2 falló"
print("  ✓ derivada_enesima n=2 OK\n")

print("=" * 40)
print("Tests de derivadas pasaron correctamente.")
print("=" * 40)
print()

# =============================================
# TESTS DE INTEGRALES
# =============================================

# --- Integral conocida: ∫₀¹ x² dx = 1/3 ---

def h_cuad(x):
    return x**2

exacta_cuad = 1.0/3.0

# Test trapecio
trap = fc.trapecio(h_cuad, 0, 1, 1000)
print(f"trapecio(x², 0, 1, n=1000) = {trap:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
print(f"  Error: {abs(trap - exacta_cuad):.2e}")
assert abs(trap - exacta_cuad) < 1e-6, "trapecio falló para x²"
print("  ✓ trapecio OK\n")

# Test simpson
simp = fc.simpson(h_cuad, 0, 1, 1000)
print(f"simpson(x², 0, 1, n=1000) = {simp:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
print(f"  Error: {abs(simp - exacta_cuad):.2e}")
assert abs(simp - exacta_cuad) < 1e-10, "simpson falló para x²"
print("  ✓ simpson OK\n")

# Test montecarlo
np.random.seed(42)
mc = fc.montecarlo(h_cuad, 0, 1, 100000)
print(f"montecarlo(x², 0, 1, N=100000) = {mc:.6f}")
print(f"  Valor exacto: {exacta_cuad:.6f}")
print(f"  Error: {abs(mc - exacta_cuad):.2e}")
assert abs(mc - exacta_cuad) < 1e-2, "montecarlo falló para x²"
print("  ✓ montecarlo OK\n")

# --- Integral conocida: ∫₀^π sin(x) dx = 2 ---

exacta_sin = 2.0

trap_sin = fc.trapecio(np.sin, 0, np.pi, 1000)
print(f"trapecio(sin, 0, π, n=1000) = {trap_sin:.10f}")
print(f"  Valor exacto: {exacta_sin}")
print(f"  Error: {abs(trap_sin - exacta_sin):.2e}")
assert abs(trap_sin - exacta_sin) < 1e-5, "trapecio falló para sin(x)"
print("  ✓ trapecio con sin(x) OK\n")

simp_sin = fc.simpson(np.sin, 0, np.pi, 1000)
print(f"simpson(sin, 0, π, n=1000) = {simp_sin:.10f}")
print(f"  Valor exacto: {exacta_sin}")
print(f"  Error: {abs(simp_sin - exacta_sin):.2e}")
assert abs(simp_sin - exacta_sin) < 1e-10, "simpson falló para sin(x)"
print("  ✓ simpson con sin(x) OK\n")

# Test simpson con n impar (debe ajustar a par automáticamente)
simp_impar = fc.simpson(h_cuad, 0, 1, 999)
print(f"simpson(x², 0, 1, n=999→1000) = {simp_impar:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
assert abs(simp_impar - exacta_cuad) < 1e-10, "simpson con n impar falló"
print("  ✓ simpson con n impar OK\n")

# =============================================
# TESTS DE CAMBIO DE VARIABLE (integrales al ∞)
# =============================================

# Test cambio_variable: z/(1-z)
assert fc.cambio_variable(0) == 0.0, "cambio_variable(0) falló"
assert fc.cambio_variable(0.5) == 1.0, "cambio_variable(0.5) falló"
assert abs(fc.cambio_variable(0.75) - 3.0) < 1e-10, "cambio_variable(0.75) falló"
print("cambio_variable(0)=0, cambio_variable(0.5)=1, cambio_variable(0.75)=3")
print("  ✓ cambio_variable OK\n")

# Test jacobiano: 1/(1-z)²
assert fc.jacobiano(0) == 1.0, "jacobiano(0) falló"
assert fc.jacobiano(0.5) == 4.0, "jacobiano(0.5) falló"
assert abs(fc.jacobiano(0.75) - 16.0) < 1e-10, "jacobiano(0.75) falló"
print("jacobiano(0)=1, jacobiano(0.5)=4, jacobiano(0.75)=16")
print("  ✓ jacobiano OK\n")

# Test integrando_z con e^(-x): ∫₀^∞ e^(-x) dx = 1
def exp_neg(x):
    return np.exp(-x)

# integrando_transformado debe dar f(z/(1-z)) * 1/(1-z)²
z_test = 0.5
iz = fc.integrando_transformado(exp_neg, z_test)
esperado_iz = np.exp(-1.0) * 4.0  # e^(-1) * jacobiano(0.5)
print(f"integrando_transformado(e^(-x), z=0.5) = {iz:.10f}")
print(f"  Valor esperado: {esperado_iz:.10f}")
assert abs(iz - esperado_iz) < 1e-10, "integrando_transformado falló"
print("  ✓ integrando_transformado OK\n")

# Test integral impropia completa: ∫₀^∞ e^(-x) dx = 1
# Usando simpson con cambio de variable en [ε, 1-ε]
eps = 1e-6
I_inf = fc.simpson(lambda z: fc.integrando_transformado(exp_neg, z), eps, 1 - eps, 1000)
print(f"∫₀^∞ e^(-x) dx (Simpson + cambio var) = {I_inf:.10f}")
print(f"  Valor exacto: 1.0")
print(f"  Error: {abs(I_inf - 1.0):.2e}")
assert abs(I_inf - 1.0) < 1e-4, "Integral impropia falló"
print("  ✓ Integral impropia OK\n")

# =============================================
# TESTS DE integral_impropia (función general)
# =============================================

# Test 1: ∫₀^∞ e^(-x) dx = 1  (reproduce el test anterior con la nueva función)
I1 = fc.integral_impropia(lambda x: np.exp(-x), 0, np.inf)
print(f"integral_impropia(e^(-x), 0, ∞) = {I1:.10f}")
print(f"  Valor exacto: 1.0")
print(f"  Error: {abs(I1 - 1.0):.2e}")
assert abs(I1 - 1.0) < 1e-4, "integral_impropia [0, ∞) falló"
print("  ✓ integral_impropia [0, ∞) OK\n")

# Test 2: ∫₋∞^0 e^(x) dx = 1
I2 = fc.integral_impropia(lambda x: np.exp(x), -np.inf, 0)
print(f"integral_impropia(e^(x), -∞, 0) = {I2:.10f}")
print(f"  Valor exacto: 1.0")
print(f"  Error: {abs(I2 - 1.0):.2e}")
assert abs(I2 - 1.0) < 1e-4, "integral_impropia (-∞, 0] falló"
print("  ✓ integral_impropia (-∞, 0] OK\n")

# Test 3: ∫₋∞^∞ e^(-x²) dx = √π
I3 = fc.integral_impropia(lambda x: np.exp(-x**2), -np.inf, np.inf, n=2000)
exacto3 = np.sqrt(np.pi)
print(f"integral_impropia(e^(-x²), -∞, ∞) = {I3:.10f}")
print(f"  Valor exacto (√π): {exacto3:.10f}")
print(f"  Error: {abs(I3 - exacto3):.2e}")
assert abs(I3 - exacto3) < 1e-3, "integral_impropia (-∞, ∞) gaussiana falló"
print("  ✓ integral_impropia (-∞, ∞) gaussiana OK\n")

# Test 4: ∫₅^∞ 1/(1+x²) dx = π/2 - arctan(5)
I4 = fc.integral_impropia(lambda x: 1/(1 + x**2), 5, np.inf)
exacto4 = np.pi/2 - np.arctan(5)
print(f"integral_impropia(1/(1+x²), 5, ∞) = {I4:.10f}")
print(f"  Valor exacto (π/2 - arctan(5)): {exacto4:.10f}")
print(f"  Error: {abs(I4 - exacto4):.2e}")
assert abs(I4 - exacto4) < 1e-4, "integral_impropia [5, ∞) falló"
print("  ✓ integral_impropia [5, ∞) OK\n")

# Test 5: ∫₋∞^-2 1/(1+x²) dx = π/2 - arctan(2)
I5 = fc.integral_impropia(lambda x: 1/(1 + x**2), -np.inf, -2)
exacto5 = np.pi/2 - np.arctan(2)
print(f"integral_impropia(1/(1+x²), -∞, -2) = {I5:.10f}")
print(f"  Valor exacto (π/2 - arctan(2)): {exacto5:.10f}")
print(f"  Error: {abs(I5 - exacto5):.2e}")
assert abs(I5 - exacto5) < 1e-4, "integral_impropia (-∞, -2] falló"
print("  ✓ integral_impropia (-∞, -2] OK\n")

# Test 6: ∫₋∞^∞ 1/(1+x²) dx = π  (Lorentziana)
I6 = fc.integral_impropia(lambda x: 1/(1 + x**2), -np.inf, np.inf, n=2000)
print(f"integral_impropia(1/(1+x²), -∞, ∞) = {I6:.10f}")
print(f"  Valor exacto (π): {np.pi:.10f}")
print(f"  Error: {abs(I6 - np.pi):.2e}")
assert abs(I6 - np.pi) < 1e-3, "integral_impropia (-∞, ∞) Lorentziana falló"
print("  ✓ integral_impropia (-∞, ∞) Lorentziana OK\n")

# Test 7: caso finito — integral_impropia debe delegar a simpson
I7 = fc.integral_impropia(lambda x: x**2, 0, 1)
print(f"integral_impropia(x², 0, 1) = {I7:.10f}")
print(f"  Valor exacto: 0.3333333333")
print(f"  Error: {abs(I7 - 1/3):.2e}")
assert abs(I7 - 1/3) < 1e-6, "integral_impropia caso finito falló"
print("  ✓ integral_impropia caso finito OK\n")

print("=" * 40)
print("Tests de integrales pasaron correctamente.")
print("=" * 40)
print()

# =============================================
# TESTS DE INTEGRADORES NUMÉRICOS (EDOs)
# =============================================

# --- Test Euler: dx/dt = -x  =>  x(t) = x₀·e^(-t) ---
# euler(f, x, h) aplica un paso: x_new = x + h*f(x)

x_euler = 1.0       # condición inicial x(0) = 1
dt = 0.001           # paso pequeño
t_final = 1.0
n_pasos = int(t_final / dt)

for _ in range(n_pasos):
    x_euler = fc.euler(lambda x: -x, x_euler, dt)

exacto_euler = np.exp(-t_final)  # e^(-1) ≈ 0.3678794...
print(f"euler(dx/dt=-x, x₀=1, t=1, dt={dt}) = {x_euler:.10f}")
print(f"  Valor exacto: {exacto_euler:.10f}")
print(f"  Error: {abs(x_euler - exacto_euler):.2e}")
assert abs(x_euler - exacto_euler) < 1e-3, "euler falló para dx/dt = -x"
print("  ✓ euler OK\n")

# --- Test Euler: dx/dt = 2x  =>  x(t) = x₀·e^(2t) ---
x_euler2 = 1.0
t_final2 = 0.5
n_pasos2 = int(t_final2 / dt)

for _ in range(n_pasos2):
    x_euler2 = fc.euler(lambda x: 2*x, x_euler2, dt)

exacto_euler2 = np.exp(2 * t_final2)  # e^1
print(f"euler(dx/dt=2x, x₀=1, t=0.5, dt={dt}) = {x_euler2:.10f}")
print(f"  Valor exacto: {exacto_euler2:.10f}")
print(f"  Error: {abs(x_euler2 - exacto_euler2):.2e}")
assert abs(x_euler2 - exacto_euler2) < 1e-2, "euler falló para dx/dt = 2x"
print("  ✓ euler con crecimiento exponencial OK\n")

# --- Test RK4: dx/dt = -x  =>  x(t) = e^(-t) ---
# rk4(t, h, x, f) con f(t, x)

x_rk4 = 1.0
dt_rk4 = 0.01  # paso más grande que Euler y aún así más preciso
t = 0.0
t_final_rk4 = 1.0
n_pasos_rk4 = int(t_final_rk4 / dt_rk4)

for i in range(n_pasos_rk4):
    x_rk4 = fc.rk4(t, dt_rk4, x_rk4, lambda t, x: -x)
    t += dt_rk4

exacto_rk4 = np.exp(-t_final_rk4)
print(f"rk4(dx/dt=-x, x₀=1, t=1, dt={dt_rk4}) = {x_rk4:.10f}")
print(f"  Valor exacto: {exacto_rk4:.10f}")
print(f"  Error: {abs(x_rk4 - exacto_rk4):.2e}")
assert abs(x_rk4 - exacto_rk4) < 1e-9, "rk4 falló para dx/dt = -x"
print("  ✓ rk4 OK\n")

# --- Test RK4: sistema vectorial dx/dt = -x (2D) ---
# x = [x1, x2], dx/dt = [-x1, -2*x2]
# Solución: x1(t) = e^(-t), x2(t) = e^(-2t)

x_vec = np.array([1.0, 1.0])
t = 0.0

def f_vec(t, x):
    return np.array([-x[0], -2*x[1]])

for i in range(n_pasos_rk4):
    x_vec = fc.rk4(t, dt_rk4, x_vec, f_vec)
    t += dt_rk4

exacto_vec = np.array([np.exp(-1.0), np.exp(-2.0)])
err_vec = np.abs(x_vec - exacto_vec)
print(f"rk4 vectorial: x₁(1)={x_vec[0]:.10f}, x₂(1)={x_vec[1]:.10f}")
print(f"  Exacto:      x₁(1)={exacto_vec[0]:.10f}, x₂(1)={exacto_vec[1]:.10f}")
print(f"  Errores: {err_vec[0]:.2e}, {err_vec[1]:.2e}")
assert np.all(err_vec < 1e-8), "rk4 vectorial falló"
print("  ✓ rk4 vectorial OK\n")

# --- Test RK4 vs Euler: RK4 debe ser mucho más preciso ---
# Ambos con el mismo paso dt=0.01 para dx/dt = -x
x_e = 1.0
x_r = 1.0
dt_comp = 0.01
t = 0.0
for i in range(100):
    x_e = fc.euler(lambda x: -x, x_e, dt_comp)
    x_r = fc.rk4(t, dt_comp, x_r, lambda t, x: -x)
    t += dt_comp

err_e = abs(x_e - np.exp(-1.0))
err_r = abs(x_r - np.exp(-1.0))
print(f"Comparación (dt={dt_comp}, t=1):")
print(f"  Euler: error = {err_e:.2e}")
print(f"  RK4:   error = {err_r:.2e}")
print(f"  RK4 es {err_e/err_r:.0f}x más preciso que Euler")
assert err_r < err_e, "RK4 debería ser más preciso que Euler"
print("  ✓ RK4 más preciso que Euler OK\n")

print("=" * 40)
print("Tests de integradores numéricos pasaron correctamente.")
print("=" * 40)
print()

# =============================================
# TESTS DE MÍNIMOS CUADRADOS (polinomio grado 3)
# =============================================
import matplotlib.pyplot as plt

# Datos: polinomio cúbico conocido y = 2 - 3x + 0.5x² + 0.8x³ + ruido
np.random.seed(7)
x_datos = np.linspace(-2, 3, 20)
sigma = np.full_like(x_datos, 2.0)  # incertidumbre constante
y_exactos = 2 - 3*x_datos + 0.5*x_datos**2 + 0.8*x_datos**3
y_datos = y_exactos + np.random.normal(0, sigma)

# Construir matriz normal A (4x4) y vector b (4x1)
# Para y = a0 + a1*x + a2*x² + a3*x³
# A_ij = sum( x^(i+j) / sigma² ),  b_i = sum( y * x^i / sigma² )
w = 1.0 / sigma**2  # pesos

A = np.zeros((4, 4))
b = np.zeros(4)
for i in range(4):
    b[i] = np.sum(w * y_datos * x_datos**i)
    for j in range(4):
        A[i, j] = np.sum(w * x_datos**(i + j))

# Resolver con minimos_cuadrados
coefs = fc.minimos_cuadrados(A, b)  # [a0, a1, a2, a3]
print(f"Coeficientes ajustados: a0={coefs[0]:.4f}, a1={coefs[1]:.4f}, a2={coefs[2]:.4f}, a3={coefs[3]:.4f}")
print(f"Coeficientes reales:    a0=2.0000, a1=-3.0000, a2=0.5000, a3=0.8000")

# Evaluar ajuste
y_ajustado = coefs[0] + coefs[1]*x_datos + coefs[2]*x_datos**2 + coefs[3]*x_datos**3

# Calcular chi cuadrada
chi2 = fc.chi_cuadrada(y_datos, y_ajustado, sigma)
ndof = len(x_datos) - 4  # grados de libertad = N - parámetros
chi2_red = chi2 / ndof
print(f"χ² = {chi2:.4f}")
print(f"χ²/ndof = {chi2_red:.4f}  (ndof = {ndof})")
assert chi2_red < 3.0, "chi² reducida demasiado alta, ajuste deficiente"
print("  ✓ minimos_cuadrados OK")
print("  ✓ chi_cuadrada OK\n")

# --- Gráfica ---
x_fino = np.linspace(x_datos.min() - 0.3, x_datos.max() + 0.3, 300)
y_fino = coefs[0] + coefs[1]*x_fino + coefs[2]*x_fino**2 + coefs[3]*x_fino**3

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x_datos, y_datos, yerr=sigma, fmt='o', color='steelblue',
            capsize=3, label='Datos con incertidumbre')
ax.plot(x_fino, y_fino, '-', color='crimson', linewidth=2, label='Ajuste cúbico')

# Ecuación y chi² en la gráfica
ecuacion = (f"$y = {coefs[0]:.2f} + ({coefs[1]:.2f})x "
            f"+ ({coefs[2]:.2f})x^2 + ({coefs[3]:.2f})x^3$")
texto = ecuacion + f"\n$\\chi^2 = {chi2:.2f}$,  $\\chi^2/\\nu = {chi2_red:.2f}$"
ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Ajuste por Mínimos Cuadrados — Polinomio de grado 3')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_minimos_cuadrados.png', dpi=150)
plt.show()
print("  Gráfica guardada en test_minimos_cuadrados.png\n")

print("=" * 40)
print("Tests de mínimos cuadrados pasaron correctamente.")
print("=" * 40)
print()

# =============================================
# TESTS DE NEWTON-RAPHSON
# =============================================

# Test 1: raíz de x² - 4 = 0  →  x = ±2
def f_nr1(x):
    return x**2 - 4

raiz1 = fc.newton_raphson(f_nr1, x=3.0, dx=1e-6, eps=1e-10, Nmax=100)
print(f"newton_raphson(x²-4, x₀=3) = {raiz1:.10f}")
print(f"  Valor exacto: 2.0")
print(f"  Error: {abs(raiz1 - 2.0):.2e}")
assert abs(raiz1 - 2.0) < 1e-8, "newton_raphson falló para x²-4 (raíz positiva)"
print("  ✓ newton_raphson raíz de x²-4 OK\n")

# Test 2: raíz negativa, partiendo de x₀ = -1
raiz1b = fc.newton_raphson(f_nr1, x=-1.0, dx=1e-6, eps=1e-10, Nmax=100)
print(f"newton_raphson(x²-4, x₀=-1) = {raiz1b:.10f}")
print(f"  Valor exacto: -2.0")
print(f"  Error: {abs(raiz1b - (-2.0)):.2e}")
assert abs(raiz1b - (-2.0)) < 1e-8, "newton_raphson falló para x²-4 (raíz negativa)"
print("  ✓ newton_raphson raíz negativa OK\n")

# Test 3: raíz de sin(x) = 0 cerca de π
raiz2 = fc.newton_raphson(np.sin, x=3.0, dx=1e-6, eps=1e-12, Nmax=100)
print(f"newton_raphson(sin(x), x₀=3) = {raiz2:.10f}")
print(f"  Valor exacto: π = {np.pi:.10f}")
print(f"  Error: {abs(raiz2 - np.pi):.2e}")
assert abs(raiz2 - np.pi) < 1e-10, "newton_raphson falló para sin(x)"
print("  ✓ newton_raphson raíz de sin(x) OK\n")

# Test 4: raíz de e^x - 3 = 0  →  x = ln(3)
def f_nr3(x):
    return np.exp(x) - 3

raiz3 = fc.newton_raphson(f_nr3, x=1.0, dx=1e-6, eps=1e-10, Nmax=100)
exacto_ln3 = np.log(3)
print(f"newton_raphson(eˣ-3, x₀=1) = {raiz3:.10f}")
print(f"  Valor exacto: ln(3) = {exacto_ln3:.10f}")
print(f"  Error: {abs(raiz3 - exacto_ln3):.2e}")
assert abs(raiz3 - exacto_ln3) < 1e-8, "newton_raphson falló para eˣ-3"
print("  ✓ newton_raphson raíz de eˣ-3 OK\n")

# Test 5: raíz de x³ - x - 2 = 0  (raíz real ≈ 1.5214)
def f_nr4(x):
    return x**3 - x - 2

raiz4 = fc.newton_raphson(f_nr4, x=2.0, dx=1e-6, eps=1e-12, Nmax=100)
# Verificar que f(raiz) ≈ 0
print(f"newton_raphson(x³-x-2, x₀=2) = {raiz4:.10f}")
print(f"  f(raíz) = {f_nr4(raiz4):.2e}")
assert abs(f_nr4(raiz4)) < 1e-10, "newton_raphson falló para x³-x-2"
print("  ✓ newton_raphson raíz de x³-x-2 OK\n")

print("=" * 40)
print("Tests de Newton-Raphson pasaron correctamente.")
print("=" * 40)
print()

# =============================================
# TESTS DE DFT, IDFT, FFT e IFFT
# =============================================

# --- Señal de prueba: suma de dos senoidales ---
N = 64
n_arr = np.arange(N)
freq1, freq2 = 5, 13  # frecuencias en bins
señal = 3.0 * np.cos(2 * np.pi * freq1 * n_arr / N) + 1.5 * np.sin(2 * np.pi * freq2 * n_arr / N)

# Test DFT: comparar con np.fft.rfft
X_dft = fc.dft(señal)
X_np = np.fft.rfft(señal)
err_dft = np.max(np.abs(np.array(X_dft) - X_np))
print(f"dft vs np.fft.rfft (señal de {N} puntos)")
print(f"  Error máximo: {err_dft:.2e}")
assert err_dft < 1e-8, "dft no coincide con np.fft.rfft"
print("  ✓ dft OK\n")

# Test DFT: picos en las frecuencias correctas
magnitudes = np.abs(X_dft)
# freq1=5 debe tener amplitud N*3/2 = 96, freq2=13 debe tener amplitud N*1.5/2 = 48
pico5 = magnitudes[freq1]
pico13 = magnitudes[freq2]
print(f"  |DFT[{freq1}]| = {pico5:.2f} (esperado: {N*3.0/2:.2f})")
print(f"  |DFT[{freq2}]| = {pico13:.2f} (esperado: {N*1.5/2:.2f})")
assert abs(pico5 - N * 3.0 / 2) < 1e-8, "dft: pico en freq1 incorrecto"
assert abs(pico13 - N * 1.5 / 2) < 1e-8, "dft: pico en freq2 incorrecto"
print("  ✓ dft picos correctos OK\n")

# Test IDFT: reconstruir la señal original
señal_rec = fc.idft(X_dft)
err_idft = np.max(np.abs(np.array(señal_rec).real - señal))
print(f"idft(dft(señal)) vs señal original")
print(f"  Error máximo: {err_idft:.2e}")
assert err_idft < 1e-8, "idft no reconstruye la señal original"
# Parte imaginaria debe ser ~0
imag_max = np.max(np.abs(np.array(señal_rec).imag))
print(f"  Parte imaginaria máxima: {imag_max:.2e}")
assert imag_max < 1e-8, "idft produce parte imaginaria no despreciable"
print("  ✓ idft OK\n")

# Test FFT: comparar con np.fft.fft
X_fft = fc.fft(señal)
X_np_full = np.fft.fft(señal)
err_fft = np.max(np.abs(np.array(X_fft) - X_np_full))
print(f"fft vs np.fft.fft (señal de {N} puntos)")
print(f"  Error máximo: {err_fft:.2e}")
assert err_fft < 1e-8, "fft no coincide con np.fft.fft"
print("  ✓ fft OK\n")

# Test IFFT: reconstruir la señal original
señal_ifft = fc.ifft(X_fft)
err_ifft = np.max(np.abs(np.array(señal_ifft) - señal))
print(f"ifft(fft(señal)) vs señal original")
print(f"  Error máximo: {err_ifft:.2e}")
assert err_ifft < 1e-8, "ifft no reconstruye la señal original"
print("  ✓ ifft OK\n")

# Test FFT con señal constante: DFT de [c, c, ..., c] = [c*N, 0, 0, ..., 0]
c = 7.0
señal_cte = np.full(16, c)
X_cte = fc.fft(señal_cte)
assert abs(X_cte[0] - c * 16) < 1e-10, "fft de constante: DC incorrecto"
assert all(abs(X_cte[k]) < 1e-10 for k in range(1, 16)), "fft de constante: componentes no-DC no son cero"
print(f"fft([{c}]*16): DC={X_cte[0]:.1f}, resto ≈ 0")
print("  ✓ fft señal constante OK\n")

# Test FFT con delta: DFT de [1, 0, 0, ..., 0] = [1, 1, 1, ..., 1]
delta = np.zeros(16)
delta[0] = 1.0
X_delta = fc.fft(delta)
assert all(abs(X_delta[k] - 1.0) < 1e-10 for k in range(16)), "fft de delta incorrecta"
print("fft([1,0,...,0]) = [1,1,...,1]")
print("  ✓ fft delta OK\n")

# Test consistencia: DFT y FFT deben dar el mismo resultado (primera mitad + 1)
X_fft_arr = np.array(X_fft)
X_dft_arr = np.array(X_dft)
err_dft_fft = np.max(np.abs(X_fft_arr[:N//2+1] - X_dft_arr))
print(f"Consistencia dft vs fft (primeros N/2+1 coeficientes)")
print(f"  Error máximo: {err_dft_fft:.2e}")
assert err_dft_fft < 1e-8, "dft y fft no son consistentes"
print("  ✓ Consistencia dft/fft OK\n")

print("=" * 40)
print("Todas las pruebas pasaron correctamente.")