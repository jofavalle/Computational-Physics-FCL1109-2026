# Seminarios de investigación

Proyectos finales de investigación del curso FCL1109, la pieza central del portafolio: cada estudiante eligió un tema avanzado de física computacional, lo implementó desde cero y lo presentó formalmente (documento/presentación + código).

## [jofavalle - Integral de camino de Feynman mediante Monte Carlo cuántico y cadenas de Markov](jofavalle-integral-camino-feynman-montecarlo/)

Implementación del formalismo de integrales de camino de Feynman para obtener el estado fundamental del oscilador armónico cuántico **sin resolver la ecuación de Schrödinger**. Se discretiza el camino en una red de tiempo imaginario y se muestrea con el **algoritmo de Metropolis, es decir, una cadena de Markov con criterio de aceptación/rechazo**, hasta que la distribución de caminos converge a la del estado fundamental.

- **Técnicas:** Monte Carlo cuántico (QMC), cadenas de Markov (Metropolis-Hastings), transformación a tiempo imaginario, esquema de actualización rojo-negro vectorizado.
- **Código:** [presentacion/codigo/qmc_camino_feynman.py](jofavalle-integral-camino-feynman-montecarlo/presentacion/codigo/qmc_camino_feynman.py)
- **Presentación:** [presentacion/presentacion.pdf](jofavalle-integral-camino-feynman-montecarlo/presentacion/presentacion.pdf) (LaTeX Beamer, con figuras de convergencia de energía, función de onda y caminos muestreados)
- **Referencias:** Landau & Páez (2018), Feynman & Hibbs (1965), Metropolis et al. (1953)

## [cesarp03 - Diagrama de bifurcación del mapa logístico](cesarp03-bifurcacion-mapa-logistico-caos/)

Estudio del mapa logístico $x_{n+1} = \mu x_n(1-x_n)$ como sistema paradigmático de tránsito hacia el caos: series de tiempo para distintos valores de $\mu$, cascada de duplicación de período y diagrama de bifurcación completo (incluyendo zoom en la región de autosimilitud cerca del punto de Feigenbaum).

- **Técnicas:** iteración de mapas discretos, análisis de series de tiempo, diagramas de bifurcación, sensibilidad a condiciones iniciales.
- **Código:** [seminario.py](cesarp03-bifurcacion-mapa-logistico-caos/seminario.py) (series de tiempo), [bifurcacion.py](cesarp03-bifurcacion-mapa-logistico-caos/bifurcacion.py) (diagrama de bifurcación)
- **Documento:** [SEMINARIO.pdf](cesarp03-bifurcacion-mapa-logistico-caos/SEMINARIO.pdf)

## [aalexanderrz - Estados coherentes de Glauber del oscilador armónico](aalexanderrz-estados-coherentes-glauber/)

Simulación de la dinámica de estados coherentes de Glauber (superposiciones cuasi-clásicas de autoestados del oscilador armónico cuántico) mediante expansión en la base de polinomios de Hermite, con animación de la evolución temporal de la densidad de probabilidad $|\psi(x,t)|^2$.

- **Técnicas:** expansión en base de autoestados, recursión de polinomios de Hermite, evolución temporal unitaria, animación con `matplotlib.animation`.
- **Código:** [estados_coherentes_glauber.py](aalexanderrz-estados-coherentes-glauber/estados_coherentes_glauber.py)
- **Documento:** [AndresRivera_EstadosCoherentes.pdf](aalexanderrz-estados-coherentes-glauber/AndresRivera_EstadosCoherentes.pdf)
