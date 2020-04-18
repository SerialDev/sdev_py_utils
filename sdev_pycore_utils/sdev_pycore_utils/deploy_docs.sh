#!/bin/sh

set -e

git checkout gh-pages
git pull --unshallow origin gh-pages

git reset --hard origin/$CIRCLE_BRANCH

sphinx-apidoc -o docs/source dslib
sphinx-build docs/source .

git add -A
git commit -m "Deploy to GitHub pages $CIRCLE_SHA1 [ci skip]"
git push -f origin gh-pages