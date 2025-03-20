# Chemin du package
PACKAGE = ../bsplyne
# Dossier de sortie pour la documentation
DOC_DIR = .
# Path to logo file
LOGO = logo.png
# URL of the logo on GitHub
LOGO_URL = https://dorian210.github.io/bsplyne/logo.png

.PHONY: all clean doc

# Commande par défaut : génère la doc
all: doc

# Génération de la documentation avec pdoc
doc:
	pdoc -o $(DOC_DIR) -d numpy --logo $(LOGO_URL) --favicon $(LOGO_URL) $(PACKAGE)

# Nettoyage des fichiers générés (sans supprimer le Makefile ni le logo)
clean:
	find $(DOC_DIR) -mindepth 1 ! -name 'Makefile' ! -name $(LOGO) -exec rm -rf {} +