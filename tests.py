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
print("Todas las pruebas pasaron correctamente.")