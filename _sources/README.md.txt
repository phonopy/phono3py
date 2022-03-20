# How to write phono3py documentation

This directory contains python-sphinx documentation source.

## How to compile

```
make html
```

## Source files

* `conf.py` contains the sphinx setting confiuration.
* `*.rst` are the usual sphinx documentation source and the filenames without `.rst` are the keys to link from toctree mainly in `index.rst`.

## How to publish

Web page files are copied to `gh-pages` branch. At the phono3py github top directory,
```
git checkout gh-pages
rm -r .buildinfo .doctrees *
```

From the directory the sphinx doc is complied,
```
rsync -avh _build/ <phono3py-repository-directory>/
```

Again, at the phono3py github top directory,
```
git add .
git commit -a -m "Update documentation ..."
git push
```
