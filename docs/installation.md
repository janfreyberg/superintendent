# Installation

The simplest way to install superintendent is to use pip:


```
pip install superintendent
```

This will install superintendet alongside all the other libraries that you will
need to run it.

If you want to use keyboard shortcuts, then you will also need to configure the
`ipyevents` notebook extension. Simply run on the command line:

```
jupyter nbextension enable --py --sys-prefix ipyevents
```

If you also want to run the examples, you need three additional packages:
`requests`, `bs4` and `wordcloud`. You can install them via pip by running
`pip install superintendent[examples]`.

## Development installation

If you want to contribute to `superintendent`, you will need to install the
test dependencies as well. You can do so with
`pip install superintendent[tests,examples]`.
