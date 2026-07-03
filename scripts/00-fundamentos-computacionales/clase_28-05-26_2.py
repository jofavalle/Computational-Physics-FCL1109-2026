import numpy as np
import matplotlib.pyplot as plt
import fcl1109 as fc


# -----------------------------------------------------------------------------
# Ejercicio 1: integral eliptica completa de primera especie K(m)
# -----------------------------------------------------------------------------
# En este ejercicio calculamos numericamente la funcion
#
#     K(m) = integral_0^(pi/2) d(theta) / sqrt(1 - m sin^2(theta))
#
# que corresponde a la integral eliptica completa de primera especie. Esta es
# la definicion estandar de K(m), y es la ecuacion que usaremos para confirmar
# que la implementacion numerica es la correcta.
#
# Ademas, la comparamos contra una aproximacion de segundo grado con seis
# parametros de la forma:
#
#     K(m) ~ a0 + a1*x + a2*x^2 - (b0 + b1*x + b2*x^2) * ln(x),
#
# donde x = 1 - m. En este script calculamos esos seis parametros mediante un
# ajuste lineal usando una referencia numerica de alta precision.


def integrando_k(theta_vals, parametro_m):
	"""Integrando de la integral eliptica completa de primera especie."""
	return 1.0 / np.sqrt(1.0 - parametro_m * np.sin(theta_vals) ** 2)


def k_numerica(parametro_m, numero_subintervalos):
	"""Evalua K(m) con Simpson usando la biblioteca del curso."""
	return fc.simpson(
		lambda theta: integrando_k(theta, parametro_m),
		0.0,
		np.pi / 2.0,
		numero_subintervalos,
	)


def ajustar_coeficientes_aproximacion(m_vals, k_referencia_vals):
	"""Ajusta los coeficientes a0, a1, a2, b0, b1, b2 de la aproximacion."""
	x_vals = 1.0 - m_vals

	matriz_diseno = np.column_stack([
		np.ones_like(x_vals),
		x_vals,
		x_vals**2,
		-np.log(x_vals),
		-x_vals * np.log(x_vals),
		-(x_vals**2) * np.log(x_vals),
	])

	coeficientes, _, _, _ = np.linalg.lstsq(matriz_diseno, k_referencia_vals, rcond=None)
	return coeficientes


def k_aproximada(parametro_m, coeficientes):
	"""Aproximacion cuadratica con seis parametros para K(m)."""
	x_val = 1.0 - parametro_m
	a0, a1, a2, b0, b1, b2 = coeficientes

	return (
		a0
		+ a1 * x_val
		+ a2 * x_val**2
		- (b0 + b1 * x_val + b2 * x_val**2) * np.log(x_val)
	)


def elliptic_K(parametro_m):
	"""Evalua K(m) usando la aproximacion ajustada y evitando la singularidad en m=1."""
	parametro_m = np.asarray(parametro_m)
	parametro_m_seguro = np.clip(parametro_m, 0.0, 0.999999)
	return k_aproximada(parametro_m_seguro, coeficientes_aproximacion)


def buscar_subintervalos_minimos(m_vals, coeficientes, tolerancia, numero_maximo_subintervalos=200):
	"""Busca el menor numero par de subintervalos que cumple la tolerancia."""
	for numero_subintervalos in range(2, numero_maximo_subintervalos + 1, 2):
		k_numerica_vals = np.array([
			k_numerica(parametro_m, numero_subintervalos) for parametro_m in m_vals
		])
		k_aproximada_vals = np.array([
			k_aproximada(parametro_m, coeficientes) for parametro_m in m_vals
		])
		error_abs = np.abs(k_numerica_vals - k_aproximada_vals)

		if np.max(error_abs) <= tolerancia:
			return numero_subintervalos, k_numerica_vals, k_aproximada_vals, error_abs

	raise RuntimeError('No se encontro un numero de subintervalos que cumpla la tolerancia.')


# -----------------------------------------------------------------------------
# Parametros del ejercicio
# -----------------------------------------------------------------------------
# La tolerancia solicitada es con respecto a la aproximacion de seis
# parametros. Como esta forma aproximante es mucho mejor que la serie truncada,
# podemos trabajar en un intervalo mas amplio de valores de m.
tolerancia = 3.0e-5
m_vals = np.linspace(0.001, 0.999, 200)


# -----------------------------------------------------------------------------
# Ajuste de la aproximacion de seis parametros
# -----------------------------------------------------------------------------
# Primero generamos una referencia numerica de alta precision y luego ajustamos
# los seis parametros de la formula aproximante.
numero_subintervalos_referencia = 8000
k_referencia_vals = np.array([
	k_numerica(parametro_m, numero_subintervalos_referencia) for parametro_m in m_vals
])

coeficientes_aproximacion = ajustar_coeficientes_aproximacion(m_vals, k_referencia_vals)
a0, a1, a2, b0, b1, b2 = coeficientes_aproximacion


# -----------------------------------------------------------------------------
# Ajuste de la evaluacion numerica
# -----------------------------------------------------------------------------
# Buscamos el menor numero de subintervalos que haga que la evaluacion numerica
# concuerde con la aproximacion polinomial al nivel pedido.
numero_subintervalos, k_numerica_vals, k_aproximada_vals, error_abs = buscar_subintervalos_minimos(
	m_vals,
	coeficientes_aproximacion,
	tolerancia,
)

indice_error_max = np.argmax(error_abs)
m_error_max = m_vals[indice_error_max]
error_maximo = error_abs[indice_error_max]


# -----------------------------------------------------------------------------
# Reporte en pantalla
# -----------------------------------------------------------------------------
print('Ecuacion usada para K(m):')
print('K(m) = integral_0^(pi/2) d(theta) / sqrt(1 - m sin^2(theta))')
print('Esta corresponde a la integral eliptica completa de primera especie.')
print()

print('Aproximacion de segundo grado usada:')
print('K(m) ~ a0 + a1*x + a2*x^2 - (b0 + b1*x + b2*x^2) * ln(x), con x = 1 - m')
print('Los coeficientes se obtienen aqui por ajuste numerico de alta precision.')
print(f'a0 = {a0:.10f}')
print(f'a1 = {a1:.10f}')
print(f'a2 = {a2:.10f}')
print(f'b0 = {b0:.10f}')
print(f'b1 = {b1:.10f}')
print(f'b2 = {b2:.10f}')
print()

print(f'Intervalo de comparacion: 0 <= m <= {m_vals[-1]:.2f}')
print(f'Tolerancia objetivo: {tolerancia:.1e}')
print(f'Subintervalos minimos encontrados con Simpson: {numero_subintervalos}')
print(f'Error maximo |K_numerica - K_aproximada| = {error_maximo:.6e}')
print(f'El error maximo ocurre en m = {m_error_max:.5f}')
print()

print('Muestras de verificacion:')
for parametro_m in [0.00, 0.25, 0.50, 0.75, 0.99]:
	valor_numerico = k_numerica(parametro_m, numero_subintervalos)
	valor_aproximado = k_aproximada(parametro_m, coeficientes_aproximacion)
	diferencia = abs(valor_numerico - valor_aproximado)
	print(
		f'm = {parametro_m:0.2f} | '
		f'K_numerica = {valor_numerico:.10f} | '
		f'K_aproximada = {valor_aproximado:.10f} | '
		f'error = {diferencia:.6e}'
	)


# -----------------------------------------------------------------------------
# Visualizacion
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(m_vals, k_numerica_vals, color='navy', lw=2.0, label='Integral numerica')
axes[0].plot(m_vals, k_aproximada_vals, color='darkorange', lw=2.0, linestyle='--', label='Aprox. de 6 parametros')
axes[0].set_xlabel('m')
axes[0].set_ylabel('K(m)')
axes[0].set_title('Integral eliptica completa de primera especie')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(m_vals, error_abs, color='darkred', lw=2.0)
axes[1].axhline(tolerancia, color='black', linestyle='--', lw=1.2, label='Tolerancia')
axes[1].set_xlabel('m')
axes[1].set_ylabel(r'$|K_{num} - K_{aprox}|$')
axes[1].set_title('Error absoluto de la aproximacion')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle('Ejercicio 1: comparacion entre K(m) numerica y aproximada', fontsize=13)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Ejercicio 2: potencial electrostatico Phi(z) usando K(m)
# -----------------------------------------------------------------------------
# Ahora reutilizamos la integral eliptica para evaluar una funcion potencial
# sobre el eje z. La expresion dada es
#
#     Phi(z) = V/2 * [1 - (k z)/(pi a) * K(k^2)],
#
# con
#
#     k = 2a / sqrt(z^2 + 4a^2).


def Phi(z, V=1.0, a=1.0):
	"""Calcula el potencial Phi(z) usando la aproximacion para K(m)."""
	k_val = 2.0 * a / np.sqrt(z**2 + 4.0 * a**2)
	K_val = elliptic_K(k_val**2)
	return V / 2.0 * (1.0 - (k_val * z) / (np.pi * a) * K_val)


z_vals = np.linspace(0.05, 10.0, 3000)
phi_vals = []

for z_val in z_vals:
	phi_vals.append(Phi(z_val, V=1.0, a=1.0))

phi_vals = np.array(phi_vals)
constante_1_r = z_vals[-1] * phi_vals[-1]
phi_ref_1_r = constante_1_r / z_vals
producto_z_phi = z_vals * phi_vals

print()
print('Ejercicio 2: potencial Phi(z)')
print(f'Phi(z_min = {z_vals[0]:.2f}) = {phi_vals[0]:.10f}')
print(f'Phi(z_max = {z_vals[-1]:.2f}) = {phi_vals[-1]:.10f}')
print(f'Valor minimo de Phi(z) en la malla = {phi_vals.min():.10f}')
print(f'Valor maximo de Phi(z) en la malla = {phi_vals.max():.10f}')
print(f'Constante de referencia para comparacion 1/r: C = z_max * Phi(z_max) = {constante_1_r:.10f}')
print(f'Valor final de z * Phi(z) = {producto_z_phi[-1]:.10f}')


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(z_vals, phi_vals, color='darkgreen', lw=2.0, label=r'$\Phi(z)$')
axes[0].plot(z_vals, phi_ref_1_r, color='darkorange', lw=2.0, linestyle='--', label=r'$C/z$')
axes[0].set_xlabel('z')
axes[0].set_ylabel(r'$\Phi(z)$')
axes[0].set_title('Comparacion del potencial con 1/r')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(z_vals, np.gradient(phi_vals, z_vals), color='darkslateblue', lw=2.0)
axes[1].set_xlabel('z')
axes[1].set_ylabel(r'$d\Phi/dz$')
axes[1].set_title('Gradiente numerico del potencial')
axes[1].grid(True, alpha=0.3)

axes[2].plot(z_vals, producto_z_phi, color='firebrick', lw=2.0)
axes[2].axhline(constante_1_r, color='black', linestyle='--', lw=1.2, label=r'$C$')
axes[2].set_xlabel('z')
axes[2].set_ylabel(r'$z\,\Phi(z)$')
axes[2].set_title('Chequeo de comportamiento 1/r')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

fig.suptitle('Ejercicio 2: uso fisico de K(m) en el potencial Phi(z)', fontsize=13)
plt.tight_layout()
plt.show()
