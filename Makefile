

gh-pages:
	git checkout gh-pages
	rm -rf *
	git checkout master docs superintendent README.md examples.ipynb
	# covert example notebook and place it in docs
	rm docs/examples.md
	jupyter nbconvert examples.ipynb --to markdown
	mv examples.md docs/examples.md
	make -C docs/ api html
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
