# cmpbayes


This is a small Python library that provides tools for performing Bayesian data
analysis on the results of running algorithms.


For now, cmpbayes provides two models:

- `Calvo` builds and samples a Bayesian model that can be used to compare
  multiple algorithms that were run on multiple problem instances. The model is
  based on the Plackett-Luce model as described by the 2018 article by Calvo et
  al., *Bayesian Inference for Algorithm Ranking Analysis*. See also the
  [corresponding R package](https://github.com/b0rxa/scmamp) by that article's
  authors.
- `Kruschke` builds and samples a Bayesian model that can be used to make
  statistical statements about the difference between two algorithms when
  run multiple times *independently* on a task. It is based on the model
  described in the 2013 article by Kruschke, *Bayesian Estimation Supersedes the
  t Test*.

  Note that this model assumes that the data points for each algorithm are be
  i.i.d.  This entails that the model does *not* take into account the
  correlation induced by cross-validation or similar methods.
- `BayesCorrTTest` builds and samples a Bayesian model that can be used to make
  statistical statements about the difference between two algorithms when each
  of those is evaluated using cross validation. It is based on the model described
  in the 2015 article by Corani and Benavoli, *A Bayesian approach for comparing
  cross-validated algorithms on multiple data sets* (in that publication it's
  called the *Bayesian t test for correlated observations model* equation (8)).



*It is strongly recommended to read about the models and their assumptions in
the respective papers before using them for making decisions.*


## Usage


Using this library can be as simple as
```Python
import cmpbayes
model = Calvo(metrics).fit()
```


We recommend to analyse the result using [the arviz
library](https://python.arviz.org/en/latest/), e.g.
```Python
azd = az.from_pystan(
    posterior=self.fit_,
    posterior_model=self.model_,
    observed_data=["orders"],
    log_likelihood={"weights": "loglik"},
)
az.plot_forest(azd, var_names=["~loglik", "~rest"])
…
```


## Running the examples


Rudimentary usage and analysis examples can be found by running the
`src/cmpbayes/__init__.py` (however, make sure to read the docstrings of the
`_analyse` methods before using them as they are for your).

You can enter a development environment by running

```bash
nix develop
```

After that,

```bash
python src/cmpbayes/__init__.py
```

runs the example.


## Installation


You should be able to install the package via pip like so:

```bash
pip install git+https://github.com/dpaetzel/cmpbayes
```


## Other libraries to check out


See [the baycomp library](https://github.com/janezd/baycomp) for comparing

- two algorithms on a single data set using a different Bayesian model than
  Kruschke
- two algorithms across multiple data sets using a Bayesian model (taking into
  account the correlation induced by cross-validation).

