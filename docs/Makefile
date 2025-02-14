# Chemin du package
PACKAGE = ../bsplyne
# Dossier de sortie pour la documentation
DOC_DIR = .
# Path to logo file
LOGO = ../logo.png

.PHONY: all clean doc

# Commande par défaut : génère la doc
all: doc

# Génération de la documentation avec pdoc
doc:
	pdoc -o $(DOC_DIR) -d numpy --logo $(LOGO) $(PACKAGE)

# Nettoyage des fichiers générés (sans supprimer le Makefile)
clean:
	find $(DOC_DIR) -mindepth 1 ! -name 'Makefile' -exec rm -rf {} +