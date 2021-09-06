# Installation

The simplest way to install superintendent is to use pip:


```
pip install superintendent
```

You will also want to install libraries that provide the UI for labelling. One
that was specifically designed to work with superintendent is
[`ipyannotations`](https://ipyannotations.readthedocs.io/en/latest/).

If you also want to run the examples, you need some dependencies not installed
by default. You can get them by installing the additional `dependencies`
example `pip install superintendent[examples]`.

## Development installation

If you want to contribute to superintendent, you can get an editable
installation of the library by using `flit`, the package used for developing
superintendent:

```
git clone https://github.com/janfreyberg/superintendent.git
cd superintendent
pip install flit
flit install --symlink --deps all
```
