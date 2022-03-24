# cmpbayes


This is a small Python library that provides tools for performing Bayesian data
analysis on the results of running algorithms.


For now, cmpbayes provides `fit_calvo` which builds and samples a Bayesian model
that can be used to compare multiple algorithms that were run on multiple
problem instances. The model is based on the Plackett-Luce model as described by
the 2018 article by Calvo et al., *Bayesian Inference for Algorithm Ranking
Analysis*. See also the [corresponding R
package](https://github.com/b0rxa/scmamp) by that article's authors.


A rudimentary usage and analysis example can be found at the end of
`src/cmpbayes/__init__.py`. You can enter a development environment by running

```bash
nix develop
```

After that,

```bash
python src/cmpbayes/__init__.py
```

runs the example.


For comparing two algorithms on a single data set as well as two algorithms on
multiple data sets, please be referred to [the baycomp
library](https://github.com/janezd/baycomp).


## Installation


You should be able to install the package via pip like so:

```bash
pip install git+https://github.com/dpaetzel/cmpbayes
```
