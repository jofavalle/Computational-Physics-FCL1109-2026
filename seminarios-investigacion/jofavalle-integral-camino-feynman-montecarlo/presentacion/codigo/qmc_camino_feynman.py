"""
Integral de camino de Feynman por Monte Carlo cuántico (algoritmo de Metropolis).

Seminario de Física Computacional (FCL1109) - Universidad de El Salvador.
Tema: Feynman Path Integral Quantum Mechanics.
Bibliografía principal: Landau & Páez, Computational Problems for Physics (2018),
sección 6.11 y listado 6.24 (programa QMC.py).

Idea física
-----------
La amplitud de propagación se escribe como una suma sobre todos los caminos
ponderados por la acción:

    G(b, a) = sum_caminos exp( i S[b,a] / hbar ),   S = integral de L dt,  L = T - V.

Pasando a tiempo imaginario (t -> -i tau) la exponencial deja de oscilar y se
convierte en un peso de Boltzmann exp(-S_E). Para tiempos imaginarios grandes
el peso queda dominado por el estado fundamental, de modo que el histograma de
las posiciones visitadas reproduce la densidad de probabilidad del estado base:

    |psi_0(x)|^2  =  (1/Z) lim_{tau->inf} integral dx_1 ... dx_{N-1} exp(-eps E).

Este script discretiza el tiempo imaginario en una red de N eslabones y usa el
algoritmo de Metropolis (sección 7.4 del libro) para muestrear los caminos.

Sistema de prueba: oscilador armónico unidimensional en unidades naturales
(hbar = m = omega = 1). El resultado exacto que sirve de comparación es

    |psi_0(x)|^2 = (1/sqrt(pi)) exp(-x^2),     E_0 = hbar*omega/2 = 0.5,
    <x^2> = hbar/(2 m omega) = 0.5.

Adaptación respecto al listado 6.24 del libro
---------------------------------------------
- Sin VPython: se usa Matplotlib con backend no interactivo y se guardan PDFs.
- El muestreo de Metropolis se vectoriza con el esquema rojo-negro (se actualizan
  primero los eslabones pares y luego los impares), válido porque el acoplamiento
  solo une vecinos contiguos. Así cada barrido completo es una operación de numpy.
- Se implementa el muestreo aquí mismo (fcl1109.py no incluye Metropolis).

Autor: José Francisco Argueta Valle.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Parámetros físicos y numéricos
# ----------------------------------------------------------------------------
N_eslabones = 200      # número de cortes de tiempo imaginario (red periódica)
eps = 0.25             # paso de tiempo imaginario [adimensional]; beta = N*eps
omega = 1.0            # frecuencia del oscilador armónico [adimensional]
masa = 1.0             # masa de la partícula [adimensional]

paso_propuesta = 1.0   # amplitud máxima del desplazamiento propuesto en Metropolis
semilla = 2026         # semilla del generador aleatorio (reproducibilidad)

barridos_termalizacion = 5000    # barridos descartados hasta alcanzar equilibrio
barridos_medicion = 40000        # barridos usados para acumular estadística

# Carpeta de salida: ../figuras respecto a este archivo
directorio_figuras = os.path.join(os.path.dirname(__file__), "..", "figuras")
os.makedirs(directorio_figuras, exist_ok=True)


# ----------------------------------------------------------------------------
# Acción euclidiana discretizada
# ----------------------------------------------------------------------------
def energia_total(camino, eps, omega, masa):
    """Acción euclidiana total del camino (red periódica).

    S_E = sum_i [ (m/2) (x_{i+1}-x_i)^2 / eps  +  eps (m omega^2 / 2) x_i^2 ].

    El primer término es la energía cinética discreta (diferencia de posiciones
    entre eslabones contiguos) y el segundo es la energía potencial del oscilador.
    Se usa solo como diagnóstico; el muestreo emplea diferencias locales de acción.
    """
    dx = np.roll(camino, -1) - camino          # x_{i+1} - x_i con frontera periódica
    cinetica = 0.5 * masa * np.sum(dx**2) / eps
    potencial = 0.5 * masa * omega**2 * eps * np.sum(camino**2)
    return cinetica + potencial


# ----------------------------------------------------------------------------
# Un barrido de Metropolis (esquema rojo-negro, vectorizado)
# ----------------------------------------------------------------------------
def barrido_metropolis(camino, eps, omega, masa, paso, hbar, rng):
    """Actualiza una vez cada eslabón del camino con el criterio de Metropolis.

    Para cada eslabón i se propone x_i -> x_i + delta (delta uniforme en
    [-paso, paso]) y se acepta el cambio si baja la acción local o, en caso
    contrario, con probabilidad exp(-Delta S / hbar). El parámetro hbar permite
    estudiar el efecto de permitir mayores fluctuaciones (ejercicio 5 del libro).

    Los eslabones pares no son vecinos entre sí, de modo que se pueden actualizar
    todos a la vez; lo mismo ocurre con los impares. Esto vuelve cada barrido una
    operación vectorial de numpy en lugar de un bucle elemento a elemento.
    """
    n = len(camino)
    aceptados = 0
    for color in (0, 1):                       # 0 = eslabones pares, 1 = impares
        idx = np.arange(color, n, 2)
        x_viejo = camino[idx]
        vecino_izq = camino[(idx - 1) % n]     # frontera periódica
        vecino_der = camino[(idx + 1) % n]
        # Desplazamiento propuesto para cada eslabón del color actual
        x_nuevo = x_viejo + paso * (2.0 * rng.random(idx.size) - 1.0)
        # Acción local (solo los términos que dependen del eslabón i)
        cin_viejo = ((x_viejo - vecino_izq)**2 + (vecino_der - x_viejo)**2)
        cin_nuevo = ((x_nuevo - vecino_izq)**2 + (vecino_der - x_nuevo)**2)
        s_viejo = 0.5 * masa * cin_viejo / eps + 0.5 * masa * omega**2 * eps * x_viejo**2
        s_nuevo = 0.5 * masa * cin_nuevo / eps + 0.5 * masa * omega**2 * eps * x_nuevo**2
        delta_s = s_nuevo - s_viejo
        # Criterio de Metropolis vectorizado
        acepta = (delta_s <= 0.0) | (rng.random(idx.size) < np.exp(-delta_s / hbar))
        camino[idx] = np.where(acepta, x_nuevo, x_viejo)
        aceptados += int(np.count_nonzero(acepta))
    return aceptados


# ----------------------------------------------------------------------------
# Simulación completa: termaliza, mide y acumula
# ----------------------------------------------------------------------------
def simular(hbar=1.0, n_instantaneas=0, bordes_hist=None, rng=None):
    """Ejecuta la cadena de Metropolis y devuelve la estadística acumulada.

    Parámetros
    ----------
    hbar : valor efectivo de hbar en el peso de Boltzmann (1.0 = caso físico).
    n_instantaneas : cuántos caminos completos guardar para graficar.
    bordes_hist : bordes del histograma de posiciones; si es None se generan.
    rng : generador aleatorio de numpy.

    Devuelve un diccionario con el histograma de |psi_0|^2, el valor de <x^2>,
    su historia acumulada (para ver la convergencia) y las instantáneas de caminos.
    """
    if rng is None:
        rng = np.random.default_rng(semilla)
    if bordes_hist is None:
        bordes_hist = np.linspace(-4.0, 4.0, 121)

    camino = np.zeros(N_eslabones)             # camino inicial: todo en x = 0
    centros = 0.5 * (bordes_hist[:-1] + bordes_hist[1:])
    histograma = np.zeros(centros.size)

    # Termalización: se descartan los primeros barridos
    aceptados = 0
    for _ in range(barridos_termalizacion):
        aceptados += barrido_metropolis(camino, eps, omega, masa, paso_propuesta, hbar, rng)
    tasa_aceptacion = aceptados / (barridos_termalizacion * N_eslabones)

    # Medición: se acumula el histograma y la media de x^2
    suma_x2 = 0.0
    cuenta = 0
    historia_x2 = np.zeros(barridos_medicion)
    instantaneas = []
    paso_guardado = max(1, barridos_medicion // max(1, n_instantaneas))

    for k in range(barridos_medicion):
        barrido_metropolis(camino, eps, omega, masa, paso_propuesta, hbar, rng)
        conteo, _ = np.histogram(camino, bins=bordes_hist)
        histograma += conteo
        suma_x2 += np.sum(camino**2)
        cuenta += camino.size
        historia_x2[k] = suma_x2 / cuenta      # estimación acumulada de <x^2>
        if n_instantaneas and (k % paso_guardado == 0) and len(instantaneas) < n_instantaneas:
            instantaneas.append(camino.copy())

    # Normalización del histograma para que integre a 1 (densidad de probabilidad)
    ancho = bordes_hist[1] - bordes_hist[0]
    densidad = histograma / (np.sum(histograma) * ancho)

    return {
        "centros": centros,
        "densidad": densidad,
        "x2_medio": historia_x2[-1],
        "historia_x2": historia_x2,
        "instantaneas": instantaneas,
        "tasa_aceptacion": tasa_aceptacion,
    }


# ----------------------------------------------------------------------------
# Curva analítica de comparación: densidad del estado fundamental del oscilador
# ----------------------------------------------------------------------------
def densidad_analitica(x, hbar=1.0):
    """|psi_0(x)|^2 = sqrt(m omega / (pi hbar)) exp(-m omega x^2 / hbar)."""
    coef = np.sqrt(masa * omega / (np.pi * hbar))
    return coef * np.exp(-masa * omega * x**2 / hbar)


# ----------------------------------------------------------------------------
# Programa principal: ejecuta las simulaciones y genera las cuatro figuras
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(semilla)

    # Simulación principal (caso físico hbar = 1), guardando algunas instantáneas
    resultado = simular(hbar=1.0, n_instantaneas=6, rng=rng)
    x = resultado["centros"]
    print("Tasa de aceptacion de Metropolis: {:.2f}".format(resultado["tasa_aceptacion"]))
    print("Energia del estado fundamental estimada (virial): "
          "E_0 = omega^2 <x^2> = {:.4f}".format(omega**2 * resultado["x2_medio"]))
    print("Valor exacto: E_0 = hbar*omega/2 = {:.4f}".format(0.5 * omega))
    print("<x^2> estimado = {:.4f}  (exacto = {:.4f})".format(resultado["x2_medio"], 0.5))

    # Figura 1: caminos espacio-temporales muestreados
    tau = np.arange(N_eslabones) * eps          # tiempo imaginario de cada eslabón
    fig1, ax1 = plt.subplots(figsize=(6, 7))
    for i, camino in enumerate(resultado["instantaneas"]):
        ax1.plot(camino, tau, lw=1.0, alpha=0.7, label="Camino muestreado" if i == 0 else None)
    ax1.axvline(0.0, color="black", lw=2.0, label="Trayectoria clásica (x = 0)")
    ax1.set_xlabel("Posición x")
    ax1.set_ylabel(r"Tiempo imaginario $\tau$")
    ax1.set_title("Caminos muestreados por Metropolis y trayectoria clásica")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(os.path.join(directorio_figuras, "caminos_espaciotemporales.pdf"))
    plt.close(fig1)

    # Figura 2: densidad del estado fundamental vs resultado analítico
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.bar(x, resultado["densidad"], width=(x[1] - x[0]), alpha=0.55,
            color="tab:blue", label="Monte Carlo (Metropolis)")
    xx = np.linspace(-4, 4, 400)
    ax2.plot(xx, densidad_analitica(xx), color="tab:red", lw=2.0,
             label=r"Exacto: $|\psi_0(x)|^2 = \pi^{-1/2} e^{-x^2}$")
    ax2.set_xlabel("Posición x")
    ax2.set_ylabel(r"$|\psi_0(x)|^2$")
    ax2.set_title("Densidad del estado fundamental del oscilador armónico")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(directorio_figuras, "funcion_onda_fundamental.pdf"))
    plt.close(fig2)

    # Figura 3: convergencia de la energía del estado fundamental
    energia_acumulada = omega**2 * resultado["historia_x2"]   # estimador del virial
    barridos = np.arange(1, barridos_medicion + 1)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(barridos, energia_acumulada, color="tab:blue", lw=1.2,
             label=r"Estimación del virial $E_0 = \omega^2\langle x^2\rangle$")
    ax3.axhline(0.5 * omega, color="tab:red", lw=2.0, ls="--",
                label=r"Valor exacto $E_0 = \hbar\omega/2 = 0.5$")
    ax3.set_xlabel("Barridos de medición")
    ax3.set_ylabel(r"$E_0$ [unidades de $\hbar\omega$]")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title("Convergencia de la energía del estado fundamental")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(directorio_figuras, "convergencia_energia.pdf"))
    plt.close(fig3)

    # Figura 4: efecto de hbar sobre las fluctuaciones (ejercicio 5)
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    colores = {0.5: "tab:green", 1.0: "tab:blue", 2.0: "tab:purple"}
    for hbar_ef in (0.5, 1.0, 2.0):
        res = simular(hbar=hbar_ef, rng=np.random.default_rng(semilla + int(10 * hbar_ef)))
        ax4.plot(res["centros"], res["densidad"], drawstyle="steps-mid",
                 color=colores[hbar_ef], lw=1.6,
                 label=r"$\hbar_{{ef}} = {:.1f}$  ($\langle x^2\rangle = {:.2f}$)".format(
                     hbar_ef, res["x2_medio"]))
    ax4.set_xlabel("Posición x")
    ax4.set_ylabel(r"$|\psi_0(x)|^2$")
    ax4.set_title("Efecto de hbar: mayor hbar permite mayores fluctuaciones")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(directorio_figuras, "efecto_hbar.pdf"))
    plt.close(fig4)

    print("Figuras guardadas en:", os.path.normpath(directorio_figuras))


if __name__ == "__main__":
    main()
