import numpy as np
# Transformacion
def x_of_z(z):
	return z/(1.0-z)
# Jacobiano dx/dz
def jacobian(z):
	return 1.0/(1.0 - z)**2
# Nueva funcion transformada en [0,1]
def transformed_integrand(f, z):
	x = x_of_z(z)
	return f(x)*jacobian(z)


