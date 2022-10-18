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
- `BimodalNonNegative` builds and samples a Bayesian model that can be used to
  make statistical statements about the difference between two algorithms when
  run multiple times *independently* on a task. Other than `Kruschke`, the units
  (i.e. the measurements) for each algorithm are assumed to be
  
  - distributed bimodally
  - non-negative

  The model uses a simple mixture consisting of two Gamma distributions. For the
  exact specifications (e.g. priors etc.), see the corresponding Stan file.

  The model was originally created to model and compare the running times of
  algorithms where a small subset of the runs was considerably faster than the
  large majority.

  Note that this model assumes that the data points for each algorithm are be
  i.i.d.  This entails that the model does *not* take into account the
  correlation induced by cross-validation or similar methods.


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
`scripts/examples.py` (however, make sure to read the docstrings of the
`_analyse` methods of the models before using them as they are for making
decisions).

After cloning the repository, you can enter a development environment by running

```bash
nix develop
```

and then run the examples by doing

```bash
python scripts/examples.py
```


## Installation


This repository is a Nix Flake, which means that you can add it rather
straightforwardly to your project's `flake.nix`:

```Nix
{
  …
  inputs.cmpbayes = "github:dpaetzel/cmpbayes";
  …
  outputs = { …, cmpbayes } : {
    …
    propagatedBuildInputs = [
      …
      cmpbayes.defaultPackage."${system}"
      …
    ];

  };
}
```


Alternatively, you should be able to install the package via pip like so:

```bash
pip install git+https://github.com/dpaetzel/cmpbayes
```


However, this is not tested so far and since we're precompiling Stan models,
this may not work out of the box. Please open an issue if this is the case for
you.


## Other libraries to check out


See [the baycomp library](https://github.com/janezd/baycomp) for comparing

- two algorithms on a single data set using a different Bayesian model than
  Kruschke
- two algorithms across multiple data sets using a Bayesian model (taking into
  account the correlation induced by cross-validation).

