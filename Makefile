

gh-pages:
	git checkout gh-pages
	rm -rf *
	git checkout master docs superintendent README.md
	# make -C docs/ api html
	make -C docs/ html
	mv ./docs/_build/html/* ./
	rm -rf docs superintendent
	echo "baseurl: /superintendent" > _config.yml
	touch .nojekyll
	git add -A
	git commit -m "publishing updated docs..."
	git push origin gh-pages
	# switch back
	git checkout master


.PHONY: gh-pages
