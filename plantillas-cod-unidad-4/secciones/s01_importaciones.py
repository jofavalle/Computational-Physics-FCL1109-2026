"""
SECCIÓN: IMPORTACIONES
======================
Patrón estándar de imports que aparece en todos los scripts de la unidad 4.

Regla de oro: importar SOLO lo que se va a usar.
En el parcial no hay fcl1109 ni scipy — solo numpy y matplotlib.
"""

# --- Bloque mínimo (onda, calor, Laplace) ---
import numpy as np
import matplotlib.pyplot as plt

# --- Con 3D (Laplace superficies, pendulums) ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # para plot_surface, add_subplot(projection='3d')

# --- Con animación (FuncAnimation) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation   # para animar una línea o imshow

# --- Combinación completa (la más frecuente en clase) ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# =============================================================================
# NOTAS RÁPIDAS
# =============================================================================
# np.linspace(a, b, N)   → N puntos equiespaciados incluyendo a y b
# np.arange(N) * dx      → N puntos: 0, dx, 2dx, ..., (N-1)*dx  (NO incluye N*dx)
# np.zeros((N, M))       → matriz NxM de ceros  (2D)
# np.zeros(N)            → vector de N ceros     (1D)
# np.ones_like(x)        → arreglo de unos con forma de x
# np.copy(u)             → copia profunda (NO alias)
# u[:]  = v[:]           → copiar en el mismo bloque de memoria (evitar reasignar)
