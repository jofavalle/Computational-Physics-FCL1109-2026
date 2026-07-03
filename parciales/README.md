# Parciales

Exámenes parciales del curso FCO4101 (60 % de la nota final).

## [Parcial 2 - Ecuación de onda 1D: modos normales](parcial-2-onda-1d-modos-normales-leapfrog/)

Solución de la ecuación de onda 1D con el esquema leapfrog (diferencias finitas centradas en tiempo y espacio), comparando la solución numérica contra la expansión analítica en modos normales (serie de Fourier). Entrega de jofavalle (AV18012).

## [Parcial 3 - Examen integrador: mecánica cuántica y electromagnetismo](parcial-3-examen-integrador-cuantica-electromagnetismo/)

Material compartido de repaso/examen con 6 temas independientes, cada uno con enunciado y solución (`simulacro_solucion.py`):

| Tema | Problema físico | Técnica numérica |
|---|---|---|
| [A - Pozo cuadrado finito](parcial-3-examen-integrador-cuantica-electromagnetismo/A-pozo-cuadrado-finito-biseccion/) | Estados ligados pares e impares en un pozo simétrico | Bisección sobre la ecuación trascendental de cuantización |
| [B - Deuterón](parcial-3-examen-integrador-cuantica-electromagnetismo/B-deuteron-estado-ligado-shooting/) | Estado ligado en un pozo nuclear (protón-neutrón) | Método de disparo + derivada logarítmica + bisección |
| [C - TDSE](parcial-3-examen-integrador-cuantica-electromagnetismo/C-tdse-paquete-gaussiano-leapfrog/) | Evolución de un paquete de onda gaussiano en un pozo infinito | Leapfrog con parte real/imaginaria separadas |
| [D - Oscilador armónico cuántico](parcial-3-examen-integrador-cuantica-electromagnetismo/D-oscilador-armonico-cuantico-rk4/) | Autoestados y autoenergías del oscilador armónico cuántico | RK4 + bisección (condición de cuantización) |
| [E - Poisson 2D](parcial-3-examen-integrador-cuantica-electromagnetismo/E-poisson-2d-cuadrupolo-sor/) | Potencial de un cuadrupolo con electrodo a potencial fijo | SOR (sobrerrelajación sucesiva) |
| [F - FDTD](parcial-3-examen-integrador-cuantica-electromagnetismo/F-fdtd-ondas-electromagneticas-1d/) | Propagación de ondas electromagnéticas 1D | FDTD (malla de Yee) |

`referencia_algoritmos_generales.py` contiene implementaciones de referencia (RK4 manual, bisección, etc.) usadas como apoyo transversal a los 6 temas.
