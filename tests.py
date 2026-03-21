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
print("Todas las pruebas pasaron correctamente.")