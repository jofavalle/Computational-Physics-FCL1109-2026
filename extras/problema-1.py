# Resolviendo un oscilador forzado amortiguado haciendo uso de la librería manteca.py

import numpy as np
import matplotlib.pyplot as plt
import manteca as mt

# --- Parámetros del oscilador ---
m = 1.0           # masa [kg]
k = 4.0           # constante del resorte [N/m]
gamma = 0.5       # coeficiente de amortiguamiento [kg/s]
F0 = 1.0          # amplitud de la fuerza externa [N]
omega_d = 1.5     # frecuencia angular de la fuerza externa [rad/s]

omega_0 = np.sqrt(k / m)   # frecuencia natural
beta = gamma / (2 * m)     # factor de amortiguamiento

# --- Condiciones iniciales ---
x0 = 1.0    # posición inicial [m]
v0 = 0.0    # velocidad inicial [m/s]

# --- Parámetros de integración ---
t0 = 0.0
tf = 50.0
dt = 0.01
N_pasos = int((tf - t0) / dt)

# --- Definición del sistema de EDOs ---
# Estado: s = [x, v]
# ds/dt = [v, -2*beta*v - omega_0^2*x + (F0/m)*cos(omega_d*t)]
def f(t, s):
    x, v = s[0], s[1]
    dxdt = v
    dvdt = -2 * beta * v - omega_0**2 * x + (F0 / m) * np.cos(omega_d * t)
    return np.array([dxdt, dvdt])

# --- Integración con RK4 ---
t_vals = np.zeros(N_pasos + 1)
x_vals = np.zeros(N_pasos + 1)
v_vals = np.zeros(N_pasos + 1)

estado = np.array([x0, v0])
t_vals[0] = t0
x_vals[0] = x0
v_vals[0] = v0

for i in range(N_pasos):
    estado = mt.rk4(t_vals[i], dt, estado, f)
    t_vals[i + 1] = t_vals[i] + dt
    x_vals[i + 1] = estado[0]
    v_vals[i + 1] = estado[1]

# --- Energía ---
E_cin = 0.5 * m * v_vals**2
E_pot = 0.5 * k * x_vals**2
E_total = E_cin + E_pot

# --- Gráficas ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Posición vs tiempo
axes[0, 0].plot(t_vals, x_vals, color="tab:blue", linewidth=0.8)
axes[0, 0].set_xlabel("t [s]")
axes[0, 0].set_ylabel("x [m]")
axes[0, 0].set_title("Posición vs tiempo")
axes[0, 0].grid(True, alpha=0.3)

# Velocidad vs tiempo
axes[0, 1].plot(t_vals, v_vals, color="tab:red", linewidth=0.8)
axes[0, 1].set_xlabel("t [s]")
axes[0, 1].set_ylabel("v [m/s]")
axes[0, 1].set_title("Velocidad vs tiempo")
axes[0, 1].grid(True, alpha=0.3)

# Espacio de fase (v vs x)
axes[1, 0].plot(x_vals, v_vals, color="tab:green", linewidth=0.5)
axes[1, 0].set_xlabel("x [m]")
axes[1, 0].set_ylabel("v [m/s]")
axes[1, 0].set_title("Espacio de fase")
axes[1, 0].grid(True, alpha=0.3)

# Energía vs tiempo
axes[1, 1].plot(t_vals, E_cin, color="tab:orange", linewidth=0.8, label="Cinética")
axes[1, 1].plot(t_vals, E_pot, color="tab:purple", linewidth=0.8, label="Potencial")
axes[1, 1].plot(t_vals, E_total, color="black", linewidth=1.0, label="Total")
axes[1, 1].set_xlabel("t [s]")
axes[1, 1].set_ylabel("E [J]")
axes[1, 1].set_title("Energía vs tiempo")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(
    f"Oscilador forzado amortiguado\n"
    f"ω₀={omega_0:.2f}, β={beta:.2f}, F₀={F0:.2f}, ω_d={omega_d:.2f}",
    fontsize=13,
)
plt.tight_layout()
plt.show()
