import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Parámetros numéricos del dominio bidimensional
# -----------------------------------------------------------------------------
# Usamos un dominio cuadrado centrado en el origen para que el cilindro
# dieléctrico quede exactamente en el centro y el campo externo uniforme sea
# fácil de imponer mediante condiciones de frontera de tipo Dirichlet.
Nx = 200
Ny = 200
Lx = 2.5
Ly = 2.5

x_vals = np.linspace(-Lx, Lx, Nx)
y_vals = np.linspace(-Ly, Ly, Ny)
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')


# -----------------------------------------------------------------------------
# Parámetros físicos del problema
# -----------------------------------------------------------------------------
# El campo externo uniforme se impone a través del potencial V = -E0*x en los
# bordes. Así, lejos del cilindro se recupera aproximadamente un campo uniforme
# dirigido en +x, ya que Ex = -dV/dx = E0.
E0 = 1.0          # [V/m] magnitud del campo externo uniforme
radio = 0.8       # [m] radio del cilindro dieléctrico
eps_r_cil = 5.0   # permitividad relativa dentro del cilindro


# -----------------------------------------------------------------------------
# Construcción del perfil dieléctrico
# -----------------------------------------------------------------------------
# Fuera del cilindro tomamos vacío (eps_r = 1). Dentro del cilindro se usa una
# permitividad relativa mayor para representar el material dieléctrico.
r = np.sqrt(X**2 + Y**2)
mascara_cilindro = r <= radio

eps_r = np.ones((Nx, Ny))
eps_r[mascara_cilindro] = eps_r_cil

print(f'dx = {dx:.4f} m, dy = {dy:.4f} m')
print(f'Puntos dentro del cilindro: {mascara_cilindro.sum()}')
print(f'Permitividad relativa interna: {eps_r_cil:.2f}')


# -----------------------------------------------------------------------------
# Funciones auxiliares para la solución numérica
# -----------------------------------------------------------------------------
def imponer_fronteras_uniformes(V, X, campo_externo):
	"""Impone V = -E0*x en todo el borde del dominio."""
	V[0, :] = -campo_externo * X[0, :]
	V[-1, :] = -campo_externo * X[-1, :]
	V[:, 0] = -campo_externo * X[:, 0]
	V[:, -1] = -campo_externo * X[:, -1]
	return V


def resolver_potencial_sor(eps_r, X, campo_externo, omega=1.85, tolerancia=1e-5, max_iter=8000):
	"""Resuelve ∇·(eps_r ∇V)=0 con Gauss-Seidel + SOR en una grilla uniforme."""
	Nx, Ny = eps_r.shape
	V = np.zeros((Nx, Ny))
	V = imponer_fronteras_uniformes(V, X, campo_externo)

	for iteracion in range(max_iter):
		V_anterior = V.copy()

		for i in range(1, Nx - 1):
			for j in range(1, Ny - 1):
				# Permitividad efectiva en los semipuntos de la celda.
				eps_este = 0.5 * (eps_r[i, j] + eps_r[i + 1, j])
				eps_oeste = 0.5 * (eps_r[i, j] + eps_r[i - 1, j])
				eps_norte = 0.5 * (eps_r[i, j] + eps_r[i, j + 1])
				eps_sur = 0.5 * (eps_r[i, j] + eps_r[i, j - 1])

				suma_eps = eps_este + eps_oeste + eps_norte + eps_sur

				V_gauss_seidel = (
					eps_este * V[i + 1, j]
					+ eps_oeste * V[i - 1, j]
					+ eps_norte * V[i, j + 1]
					+ eps_sur * V[i, j - 1]
				) / suma_eps

				V[i, j] = (1.0 - omega) * V[i, j] + omega * V_gauss_seidel

		V = imponer_fronteras_uniformes(V, X, campo_externo)
		error = np.max(np.abs(V - V_anterior))

		if error < tolerancia:
			return V, iteracion + 1, error

	return V, max_iter, error


def calcular_campo_electrico(V, dx, dy):
	"""Calcula Ex y Ey con diferencias finitas centradas."""
	Ex = np.zeros_like(V)
	Ey = np.zeros_like(V)

	Ex[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dx)
	Ey[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dy)

	# Derivadas hacia adelante/atrás en los bordes.
	Ex[0, :] = -(V[1, :] - V[0, :]) / dx
	Ex[-1, :] = -(V[-1, :] - V[-2, :]) / dx
	Ey[:, 0] = -(V[:, 1] - V[:, 0]) / dy
	Ey[:, -1] = -(V[:, -1] - V[:, -2]) / dy

	return Ex, Ey


# -----------------------------------------------------------------------------
# Solución del potencial y cálculo del campo eléctrico
# -----------------------------------------------------------------------------
# Se resuelve primero el potencial en todo el dominio. Después se calcula el
# campo eléctrico a partir del gradiente negativo del potencial.
V, iteraciones, error_final = resolver_potencial_sor(
	eps_r,
	X,
	E0,
	omega=1.85,
	tolerancia=1e-5,
	max_iter=8000,
)

Ex, Ey = calcular_campo_electrico(V, dx, dy)
magnitud_E = np.sqrt(Ex**2 + Ey**2)


# -----------------------------------------------------------------------------
# Máscaras para comparar interior y exterior del dieléctrico
# -----------------------------------------------------------------------------
# Excluimos una franja angosta alrededor de la interfaz para evitar que el
# promedio quede dominado por la discontinuidad numérica de la frontera.
mascara_interior = r <= 0.85 * radio
mascara_exterior = r >= 1.15 * radio

campo_promedio_interior = magnitud_E[mascara_interior].mean()
campo_promedio_exterior = magnitud_E[mascara_exterior].mean()

indice_centro_y = Ny // 2
potencial_eje_x = V[:, indice_centro_y]
campo_x_eje = Ex[:, indice_centro_y]

print(f'Convergencia alcanzada en {iteraciones} iteraciones')
print(f'Error máximo final = {error_final:.3e}')
print(f'|E| promedio dentro del cilindro = {campo_promedio_interior:.4f} V/m')
print(f'|E| promedio fuera del cilindro = {campo_promedio_exterior:.4f} V/m')
print(f'Relación interior/exterior = {campo_promedio_interior / campo_promedio_exterior:.4f}')


# -----------------------------------------------------------------------------
# Visualización del potencial, las equipotenciales y las líneas de campo
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: mapa del potencial y equipotenciales.
cf = axes[0, 0].contourf(x_vals, y_vals, V.T, levels=45, cmap='coolwarm')
axes[0, 0].contour(x_vals, y_vals, V.T, levels=18, colors='black', linewidths=0.5)
axes[0, 0].add_patch(plt.Circle((0.0, 0.0), radio, fill=False, color='white', linewidth=2.0))
axes[0, 0].set_title('Potencial eléctrico y equipotenciales')
axes[0, 0].set_xlabel('x [m]')
axes[0, 0].set_ylabel('y [m]')
plt.colorbar(cf, ax=axes[0, 0], label='V [V]')

# Panel 2: líneas de campo eléctrico.
strm = axes[0, 1].streamplot(
	x_vals,
	y_vals,
	Ex.T,
	Ey.T,
	color=magnitud_E.T,
	cmap='plasma',
	density=1.6,
	linewidth=1.0,
)
axes[0, 1].add_patch(plt.Circle((0.0, 0.0), radio, fill=False, color='black', linewidth=2.0))
axes[0, 1].set_title('Líneas de campo eléctrico')
axes[0, 1].set_xlabel('x [m]')
axes[0, 1].set_ylabel('y [m]')
plt.colorbar(strm.lines, ax=axes[0, 1], label='|E| [V/m]')

# Panel 3: distribución espacial de permitividad relativa.
im = axes[1, 0].contourf(x_vals, y_vals, eps_r.T, levels=20, cmap='viridis')
axes[1, 0].add_patch(plt.Circle((0.0, 0.0), radio, fill=False, color='white', linewidth=2.0))
axes[1, 0].set_title('Perfil de permitividad relativa')
axes[1, 0].set_xlabel('x [m]')
axes[1, 0].set_ylabel('y [m]')
plt.colorbar(im, ax=axes[1, 0], label=r'$\epsilon_r$')

# Panel 4: comparación del campo sobre el eje horizontal que cruza el cilindro.
axes[1, 1].plot(x_vals, campo_x_eje, color='darkred', lw=2, label=r'$E_x(x, y=0)$')
axes[1, 1].axvspan(-radio, radio, color='gray', alpha=0.25, label='Región dieléctrica')
axes[1, 1].axhline(E0, color='black', linestyle='--', lw=1.2, label='Campo externo ideal')
axes[1, 1].set_title('Comparación del campo en el eje del cilindro')
axes[1, 1].set_xlabel('x [m]')
axes[1, 1].set_ylabel(r'$E_x$ [V/m]')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

fig.suptitle('Cilindro dieléctrico en un campo uniforme', fontsize=14)
plt.tight_layout()
plt.show()
