# ============================================================================
# Makefile para compilar la presentacion (Beamer + biblatex/biber)
# Uso:
#   make          Compila la presentacion completa (con bibliografia)
#   make quick    Compilacion rapida (sin bibliografia)
#   make clean    Elimina archivos auxiliares
#   make purge    Elimina auxiliares + PDF
#   make watch    Compilacion continua (requiere latexmk)
# ============================================================================

# bash y no el /bin/sh por defecto: en Debian es dash, que no expande las llaves
# de 'clean' y deja todos los auxiliares sin borrar.
SHELL := /bin/bash

MAIN = presentacion
TEX  = pdflatex
BIB  = biber
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all quick clean purge watch

all:
	$(TEX) $(FLAGS) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(FLAGS) $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex

quick:
	$(TEX) $(FLAGS) $(MAIN).tex

clean:
	rm -f $(MAIN).{aux,bbl,bcf,blg,log,nav,out,run.xml,snm,toc,vrb,fdb_latexmk,fls,synctex.gz}

purge: clean
	rm -f $(MAIN).pdf

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex
