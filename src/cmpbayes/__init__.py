# Copyright (C) 2022 David Pätzel <david.paetzel@posteo.de>
# SPDX-License-Identifier: GPL-3.0-only

__version__ = '1.0.0-beta'

import arviz as az
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from pkg_resources import resource_filename

pl_stan_filename = resource_filename(__name__, "pl_model.stan")
kruschke_filename = resource_filename(__name__, "kruschke.stan")


def _generate_random_seed():
    return int(np.random.default_rng().integers(low=0,
                                                high=2**31,
                                                endpoint=False))


class Calvo:
    """
    A Bayesian model of the ranking of the algorithms that produced the provided
    data.

    The underlying model is described by two articles:

    - the 2018 article by Calvo et al., Bayesian Inference for Algorithm Ranking
      Analysis, and
    - the 2020 article by Calvo et al., Bayesian performance analysis for
      black-box optimization benchmarking.
    """

    def __init__(self, metrics, higher_better=True, algorithm_labels=None):
        """
        Parameters
        ----------
        metrics : array of shape (n_instances, n_algorithms)
            Each row corresponds to one problem instance (e.g. one optimization
            problem, one learning task, …). Columns correspond to algorithms.
            Entries are performance statistics values (e.g. MSE, accuracy, maximum
            fitness, average RL return, …).
        higher_better : bool
            Whether higher metrics values are better than lower ones. This is e.g.
            the case for accuracy on classification tasks but not for MSE on
            regression tasks. This is required in order to be able to correctly
            compute the algorithm ranking.
        algorithm_labels : list of str or None
            Labels for the algorithms being compared (e.g. their names) to be
            used in the `arviz.InferenceData` object created as the `data_`
            attribute. If `None`, they are numbered.
        """
        self.metrics = metrics
        self.higher_better = higher_better
        self.algorithm_labels = algorithm_labels

    def fit(self, random_seed=None, **kwargs):
        """
        Builds the model and samples from its posterior.

        Parameters
        ----------
        random_seed : non-negative int < 2**31 - 1
            Random seed to be used for sampling. See
            [`stan.build`](https://pystan.readthedocs.io/en/latest/reference.html).
        kwargs : kwargs
            Are passed through to `stan.model.Model.sample`. You may set
            `num_samples`, `num_warmup` and many more options here. See the
            documentation of
            [sample](https://pystan.readthedocs.io/en/latest/reference.html#stan.model.Model.sample)
            as well as the [code of the sampler currently
            used](https://github.com/stan-dev/stan/blob/develop/src/stan/services/sample/hmc_nuts_diag_e_adapt.hpp).

        Returns
        -------
        object
           The fitted model (`self`).
        """
        # For each instance, get the sorting permutation of the algorithms (i.e. the
        # indexes of the algorithms in the order such that their metrics are sorted
        # ascendingly).
        orders = np.argsort(self.metrics, axis=1)

        # np.argsort sorts ascendingly, if higher is better we thus want to reverse the
        # ordering such that the best stands at the front.
        if self.higher_better:
            orders = np.flip(orders, axis=1)

        # Stan starts indexes from 1 and each element of orders is an index (i.e. an
        # algorithm number).
        orders = orders + 1

        # Read the Stan model from disk.
        with open(pl_stan_filename) as f:
            program_code = f.read()

        n_instances, n_algorithms = self.metrics.shape

        data = {
            "n_instances": n_instances,
            "n_algorithms": n_algorithms,
            "orders": orders,
            "weights_instances": np.ones(n_instances),
            # Uniform prior.
            "alpha": np.ones(n_algorithms),
        }

        if random_seed is None:
            random_seed = _generate_random_seed()

        self.model_: stan.model.Model = stan.build(program_code,
                                                   data=data,
                                                   random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        self.data_: arviz.InferenceData = az.from_pystan(
            posterior=self.fit_,
            # posterior_predictive
            # predictions
            # prior
            # prior_predictive
            observed_data=["orders"],
            # constant_data
            # predictions_constant_data
            # TODO log_likelihood
            coords={
                "algorithm_labels":
                np.arange(n_algorithms)
                if self.algorithm_labels is None else self.algorithm_labels
            },
            dims={"weights": ["algorithm_labels"]},
            posterior_model=self.model_,
            # prior_model
        )

        return self

    def _analyse(self):
        """
        Perform a rudimentary analysis of the built model.

        Mainly meant to be a starting point for analysing the models built as
        well as showcase a few things that can be done with the result,
        especially when combining this library with [the arviz
        library](https://python.arviz.org/en/latest/).

        Warnings
        --------
        This is explicitely *not* meant as a best practice of how to analyse the
        results and may change any time. Read up on how to interpret models and
        especially on how to work out whether sampling even worked as intended.
        """
        df = self.fit_.to_frame()
        weights = df.filter(regex="^weights.*$")
        print(weights.mean())
        print(weights.quantile([0.05, 0.95]))

        # TODO Add ppd example, see
        # https://mc-stan.org/docs/2_24/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html
        # as well as posterior_predictive="weights_hat" option to from_pystan

        summary = az.summary(self.data_)
        print(summary)
        az.plot_forest(self.data_, var_names=["~loglik", "~rest"])
        az.plot_posterior(self.data_)
        az.plot_trace(self.data_)
        plt.show()


class Kruschke:
    """
    A Bayesian model that can be used to make statistical statements about the
    difference between two algorithms when run multiple times *independently* on
    a task.

    The underlying model is described in the 2013 article by Kruschke, *Bayesian
    Estimation Supersedes the t Test*.

    Notes
    -----
    This model assumes the data points to be paired (i.e. the first data point
    of the first algorithm corresponds to the first data point of the second
    algorithm) as well as that the data points for each algorithm are be i.i.d.
    This entails that the model does *not* take into account the correlation
    induced by cross-validation or similar methods.
    """

    def __init__(self, y1, y2):
        """
        Parameters
        ----------
        y1 : array of shape (n_tasks,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the first method to compare.
        y2 : array of shape (n_tasks,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the second method to compare.

        Notes
        -----
        As of now, this expects arrays (in particular, `pandas.DataFrame` or
        `pandas.Series` are not supported as inputs—use their `to_numpy()`
        method before passing them here).
        """
        self.y1 = y1
        self.y2 = y2

    def fit(self, random_seed=None, **kwargs):
        """
        Compares the two samples using the model described in the 2013 article by
        Kruschke, *Bayesian Estimation Supersedes the t Test*.

        Parameters
        ----------
        random_seed : non-negative int < 2**31 - 1
            Random seed to be used for sampling. See
            [`stan.build`](https://pystan.readthedocs.io/en/latest/reference.html).
        kwargs : kwargs
            Are passed through to `stan.model.Model.sample`. You may set
            `num_samples`, `num_warmup` and many more options here. See the
            documentation of
            [sample](https://pystan.readthedocs.io/en/latest/reference.html#stan.model.Model.sample)
            as well as the [code of the sampler currently
            used](https://github.com/stan-dev/stan/blob/develop/src/stan/services/sample/hmc_nuts_diag_e_adapt.hpp).

        Returns
        -------
        object
           The fitted model (`self`).
        """
        # Read the Stan model from disk.
        with open(kruschke_filename) as f:
            program_code = f.read()

        n_runs, = self.y1.shape

        data = dict(n_runs=n_runs, y1=self.y1, y2=self.y2)

        if random_seed is None:
            random_seed = _generate_random_seed()

        self.model_: stan.model.Model = stan.build(program_code,
                                                   data=data,
                                                   random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        self.data_: arviz.InferenceData = az.from_pystan(
            posterior=self.fit_,
            posterior_predictive=["y1_rep", "y2_rep"],
            # predictions
            # prior
            # prior_predictive
            observed_data=["y1", "y2"],
            # constant_data
            # predictions_constant_data
            # log_likelihood
            # coords
            # dims
            posterior_model=self.model_,
            # prior_model
        )

        return self

    # TODO Add rope here
    def _analyse(self):
        """
        Perform a rudimentary analysis of the built model.

        Mainly meant to be a starting point for analysing the models built as
        well as showcase a few things that can be done with the result,
        especially when combining this library with [the arviz
        library](https://python.arviz.org/en/latest/).

        Warnings
        --------
        This is explicitely *not* meant as a best practice of how to analyse the
        results and may change any time. Read up on how to interpret models and
        especially on how to work out whether sampling even worked as intended.
        """
        # See https://oriolabril.github.io/arviz/api/generated/arviz.from_pystan.html .
        azd = az.from_pystan(
            posterior=self.fit_,
            posterior_model=self.model_,
            observed_data=["y1", "y2"],
            posterior_predictive=["y1_rep", "y2_rep"],
        )

        var_names = ["~y2_rep", "~y1_rep"]

        summary = az.summary(azd, filter_vars="like", var_names=var_names)
        print(summary)

        az.plot_ppc(azd,
                    kind="kde",
                    data_pairs={
                        "y1": "y1_rep",
                        "y2": "y2_rep"
                    },
                    num_pp_samples=10)
        az.plot_posterior(azd, filter_vars="like", var_names=var_names)
        az.plot_trace(azd, filter_vars="like", var_names=var_names)
        az.plot_density(azd.posterior.mu2 - azd.posterior.mu1, hdi_markers="v")
        plt.show()


def _test_calvo():
    seed = 2
    rng = np.random.default_rng(seed)

    n_instances = 1000

    algorithm_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
    n_algorithms = len(algorithm_labels)

    metrics = rng.normal(loc=100, scale=40, size=(n_instances, n_algorithms))

    model = Calvo(metrics,
                  higher_better=True,
                  algorithm_labels=algorithm_labels).fit()
    # import IPython
    # IPython.embed()
    model._analyse()


def _test_kruschke():
    seed = 3
    rng = np.random.default_rng(seed)

    n_runs = 30

    y1 = rng.normal(loc=100, scale=20, size=(n_runs))
    y2 = rng.normal(loc=103, scale=10, size=(n_runs))

    model = Kruschke(y1, y2).fit()
    model._analyse()


tests = {
    "kruschke": _test_kruschke,
    "calvo": _test_calvo,
}


@click.command()
@click.argument("name")
def test(name):
    """
    Run a very simple test scenario from randomly generated data for the model
    NAME.
    """
    if name not in tests:
        click.echo("Nope, that model doesn't exist or there are no tests for "
                   "that model. Names of models with tests are: "
                   f"{list(tests.keys())}")
    else:
        tests[name]()


if __name__ == '__main__':
    test()
