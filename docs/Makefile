# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = superintendent
SOURCEDIR     = .
BUILDDIR      = _build
PROJDIR       = "../$(SPHINXPROJ)"
GH_PAGES_SOURCES = Makefile docs superintendent README.md

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

server:
	(cd _build/html && python -m http.server)

api:
	rm -f api/*
	sphinx-apidoc -o api $(PROJDIR) $(PROJDIR)/version.py $(PROJDIR)/base.py

gh-pages:
	git checkout gh-pages
	rm -rf *
	git checkout master $(GH_PAGES_SOURCES)
	git reset HEAD
	# cp ./index.ipynb doc/examples.ipynb
	make api
	make html
	mv -fv $(BUILDDIR)/html/* ./
	rm -rf $(SOURCEDIR) build
	> .nojekyll
	git add -A
	git commit -m "Generated gh-pages automatically."
	git push origin gh-pages
	git checkout master

.PHONY: help Makefile api server gh-pages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
