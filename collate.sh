#! /bin/bash
set -x
set -e

mkdir -p /tmp/scratch
rm -rf /tmp/scratch/*

find $(pwd) -iname 'master.tex' -print0 | \
    xargs -0 sh -c 'for f; do echo "$f"; cd $(dirname $f) && latexmk -C $f && latexmk -pdf $f; done'

for f in $(find . -iname 'master.pdf'); do
    cp $f /tmp/scratch/$(echo "${f:2}" | tr / - | sed -e 's/-master//')
done

