''' Script para testear el funcionamimiento de la librería manteca.py '''
import manteca as mt
import numpy as np

# --- Funciones de prueba ---

def f(x):
    return x**3  # f'(x) = 3x², f''(x) = 6x, f'''(x) = 6

def g(x):
    return np.sin(x)  # g'(x) = cos(x)

x0 = 2.0
h = 1e-5

# Test derivada_d (derivada hacia delante)
dd = mt.derivada_d(f, x0, h)
esperado_d = 3 * x0**2  # f'(2) = 12
print(f"derivada_d(x³, x={x0}, h={h}) = {dd:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(dd - esperado_d):.2e}")
assert abs(dd - esperado_d) < 1e-4, "derivada_d falló para x³"
print("  ✓ derivada_d OK\n")

# Test derivada_c (derivada central)
dc = mt.derivada_c(f, x0, h)
print(f"derivada_c(x³, x={x0}, h={h}) = {dc:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(dc - esperado_d):.2e}")
assert abs(dc - esperado_d) < 1e-8, "derivada_c falló para x³"
print("  ✓ derivada_c OK\n")

# Test derivada_c con sin(x)
dc_sin = mt.derivada_c(g, np.pi/4, h)
esperado_cos = np.cos(np.pi/4)
print(f"derivada_c(sin, x=π/4, h={h}) = {dc_sin:.10f}")
print(f"  Valor esperado: {esperado_cos:.10f}")
print(f"  Error: {abs(dc_sin - esperado_cos):.2e}")
assert abs(dc_sin - esperado_cos) < 1e-8, "derivada_c falló para sin(x)"
print("  ✓ derivada_c con sin(x) OK\n")

# Test nderiv — orden 0 (debe devolver f(x))
nd0 = mt.nderiv(f, x0, 0)
print(f"nderiv(x³, x={x0}, n=0) = {nd0}")
print(f"  Valor esperado: {f(x0)}")
assert nd0 == f(x0), "nderiv n=0 falló"
print("  ✓ nderiv n=0 OK\n")

# Test nderiv — primera derivada
nd1 = mt.nderiv(f, x0, 1)
print(f"nderiv(x³, x={x0}, n=1) = {nd1:.10f}")
print(f"  Valor esperado: {esperado_d}")
print(f"  Error: {abs(nd1 - esperado_d):.2e}")
assert abs(nd1 - esperado_d) < 1e-4, "nderiv n=1 falló"
print("  ✓ nderiv n=1 OK\n")

# Test nderiv — segunda derivada
nd2 = mt.nderiv(f, x0, 2)
esperado_2 = 6 * x0  # f''(2) = 12
print(f"nderiv(x³, x={x0}, n=2) = {nd2:.6f}")
print(f"  Valor esperado: {esperado_2}")
print(f"  Error: {abs(nd2 - esperado_2):.2e}")
assert abs(nd2 - esperado_2) < 1e-2, "nderiv n=2 falló"
print("  ✓ nderiv n=2 OK\n")

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
trap = mt.trapecio(h_cuad, 0, 1, 1000)
print(f"trapecio(x², 0, 1, n=1000) = {trap:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
print(f"  Error: {abs(trap - exacta_cuad):.2e}")
assert abs(trap - exacta_cuad) < 1e-6, "trapecio falló para x²"
print("  ✓ trapecio OK\n")

# Test simpson
simp = mt.simpson(h_cuad, 0, 1, 1000)
print(f"simpson(x², 0, 1, n=1000) = {simp:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
print(f"  Error: {abs(simp - exacta_cuad):.2e}")
assert abs(simp - exacta_cuad) < 1e-10, "simpson falló para x²"
print("  ✓ simpson OK\n")

# Test montecarlo
np.random.seed(42)
mc = mt.montecarlo(h_cuad, 0, 1, 100000)
print(f"montecarlo(x², 0, 1, N=100000) = {mc:.6f}")
print(f"  Valor exacto: {exacta_cuad:.6f}")
print(f"  Error: {abs(mc - exacta_cuad):.2e}")
assert abs(mc - exacta_cuad) < 1e-2, "montecarlo falló para x²"
print("  ✓ montecarlo OK\n")

# --- Integral conocida: ∫₀^π sin(x) dx = 2 ---

exacta_sin = 2.0

trap_sin = mt.trapecio(np.sin, 0, np.pi, 1000)
print(f"trapecio(sin, 0, π, n=1000) = {trap_sin:.10f}")
print(f"  Valor exacto: {exacta_sin}")
print(f"  Error: {abs(trap_sin - exacta_sin):.2e}")
assert abs(trap_sin - exacta_sin) < 1e-5, "trapecio falló para sin(x)"
print("  ✓ trapecio con sin(x) OK\n")

simp_sin = mt.simpson(np.sin, 0, np.pi, 1000)
print(f"simpson(sin, 0, π, n=1000) = {simp_sin:.10f}")
print(f"  Valor exacto: {exacta_sin}")
print(f"  Error: {abs(simp_sin - exacta_sin):.2e}")
assert abs(simp_sin - exacta_sin) < 1e-10, "simpson falló para sin(x)"
print("  ✓ simpson con sin(x) OK\n")

# Test simpson con n impar (debe ajustar a par automáticamente)
simp_impar = mt.simpson(h_cuad, 0, 1, 999)
print(f"simpson(x², 0, 1, n=999→1000) = {simp_impar:.10f}")
print(f"  Valor exacto: {exacta_cuad:.10f}")
assert abs(simp_impar - exacta_cuad) < 1e-10, "simpson con n impar falló"
print("  ✓ simpson con n impar OK\n")

# =============================================
# TESTS DE CAMBIO DE VARIABLE (integrales al ∞)
# =============================================

# Test x_of_z: z/(1-z)
assert mt.x_of_z(0) == 0.0, "x_of_z(0) falló"
assert mt.x_of_z(0.5) == 1.0, "x_of_z(0.5) falló"
assert abs(mt.x_of_z(0.75) - 3.0) < 1e-10, "x_of_z(0.75) falló"
print("x_of_z(0)=0, x_of_z(0.5)=1, x_of_z(0.75)=3")
print("  ✓ x_of_z OK\n")

# Test jacobian: 1/(1-z)²
assert mt.jacobian(0) == 1.0, "jacobian(0) falló"
assert mt.jacobian(0.5) == 4.0, "jacobian(0.5) falló"
assert abs(mt.jacobian(0.75) - 16.0) < 1e-10, "jacobian(0.75) falló"
print("jacobian(0)=1, jacobian(0.5)=4, jacobian(0.75)=16")
print("  ✓ jacobian OK\n")

# Test integrando_z con e^(-x): ∫₀^∞ e^(-x) dx = 1
def exp_neg(x):
    return np.exp(-x)

# integrando_z debe dar f(z/(1-z)) * 1/(1-z)²
z_test = 0.5
iz = mt.integrando_z(exp_neg, z_test)
esperado_iz = np.exp(-1.0) * 4.0  # e^(-1) * jacobian(0.5)
print(f"integrando_z(e^(-x), z=0.5) = {iz:.10f}")
print(f"  Valor esperado: {esperado_iz:.10f}")
assert abs(iz - esperado_iz) < 1e-10, "integrando_z falló"
print("  ✓ integrando_z OK\n")

# Test integral impropia completa: ∫₀^∞ e^(-x) dx = 1
# Usando simpson con cambio de variable en [ε, 1-ε]
eps = 1e-6
I_inf = mt.simpson(lambda z: mt.integrando_z(exp_neg, z), eps, 1 - eps, 1000)
print(f"∫₀^∞ e^(-x) dx (Simpson + cambio var) = {I_inf:.10f}")
print(f"  Valor exacto: 1.0")
print(f"  Error: {abs(I_inf - 1.0):.2e}")
assert abs(I_inf - 1.0) < 1e-4, "Integral impropia falló"
print("  ✓ Integral impropia OK\n")

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
    x_euler = mt.euler(lambda x: -x, x_euler, dt)

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
    x_euler2 = mt.euler(lambda x: 2*x, x_euler2, dt)

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
    x_rk4 = mt.rk4(t, dt_rk4, x_rk4, lambda t, x: -x)
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
    x_vec = mt.rk4(t, dt_rk4, x_vec, f_vec)
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
    x_e = mt.euler(lambda x: -x, x_e, dt_comp)
    x_r = mt.rk4(t, dt_comp, x_r, lambda t, x: -x)
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
coefs = mt.minimos_cuadrados(A, b)  # [a0, a1, a2, a3]
print(f"Coeficientes ajustados: a0={coefs[0]:.4f}, a1={coefs[1]:.4f}, a2={coefs[2]:.4f}, a3={coefs[3]:.4f}")
print(f"Coeficientes reales:    a0=2.0000, a1=-3.0000, a2=0.5000, a3=0.8000")

# Evaluar ajuste
y_ajustado = coefs[0] + coefs[1]*x_datos + coefs[2]*x_datos**2 + coefs[3]*x_datos**3

# Calcular chi cuadrada
chi2 = mt.chi_cuadrada(y_datos, y_ajustado, sigma)
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

raiz1 = mt.newtonr(f_nr1, x=3.0, dx=1e-6, eps=1e-10, Nmax=100)
print(f"newtonr(x²-4, x₀=3) = {raiz1:.10f}")
print(f"  Valor exacto: 2.0")
print(f"  Error: {abs(raiz1 - 2.0):.2e}")
assert abs(raiz1 - 2.0) < 1e-8, "newtonr falló para x²-4 (raíz positiva)"
print("  ✓ newtonr raíz de x²-4 OK\n")

# Test 2: raíz negativa, partiendo de x₀ = -1
raiz1b = mt.newtonr(f_nr1, x=-1.0, dx=1e-6, eps=1e-10, Nmax=100)
print(f"newtonr(x²-4, x₀=-1) = {raiz1b:.10f}")
print(f"  Valor exacto: -2.0")
print(f"  Error: {abs(raiz1b - (-2.0)):.2e}")
assert abs(raiz1b - (-2.0)) < 1e-8, "newtonr falló para x²-4 (raíz negativa)"
print("  ✓ newtonr raíz negativa OK\n")

# Test 3: raíz de sin(x) = 0 cerca de π
raiz2 = mt.newtonr(np.sin, x=3.0, dx=1e-6, eps=1e-12, Nmax=100)
print(f"newtonr(sin(x), x₀=3) = {raiz2:.10f}")
print(f"  Valor exacto: π = {np.pi:.10f}")
print(f"  Error: {abs(raiz2 - np.pi):.2e}")
assert abs(raiz2 - np.pi) < 1e-10, "newtonr falló para sin(x)"
print("  ✓ newtonr raíz de sin(x) OK\n")

# Test 4: raíz de e^x - 3 = 0  →  x = ln(3)
def f_nr3(x):
    return np.exp(x) - 3

raiz3 = mt.newtonr(f_nr3, x=1.0, dx=1e-6, eps=1e-10, Nmax=100)
exacto_ln3 = np.log(3)
print(f"newtonr(eˣ-3, x₀=1) = {raiz3:.10f}")
print(f"  Valor exacto: ln(3) = {exacto_ln3:.10f}")
print(f"  Error: {abs(raiz3 - exacto_ln3):.2e}")
assert abs(raiz3 - exacto_ln3) < 1e-8, "newtonr falló para eˣ-3"
print("  ✓ newtonr raíz de eˣ-3 OK\n")

# Test 5: raíz de x³ - x - 2 = 0  (raíz real ≈ 1.5214)
def f_nr4(x):
    return x**3 - x - 2

raiz4 = mt.newtonr(f_nr4, x=2.0, dx=1e-6, eps=1e-12, Nmax=100)
# Verificar que f(raiz) ≈ 0
print(f"newtonr(x³-x-2, x₀=2) = {raiz4:.10f}")
print(f"  f(raíz) = {f_nr4(raiz4):.2e}")
assert abs(f_nr4(raiz4)) < 1e-10, "newtonr falló para x³-x-2"
print("  ✓ newtonr raíz de x³-x-2 OK\n")

print("=" * 40)
print("Todas las pruebas pasaron correctamente.")