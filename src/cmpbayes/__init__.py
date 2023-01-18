# Copyright (C) 2022 David Pätzel <david.paetzel@posteo.de>
# SPDX-License-Identifier: GPL-3.0-only

__version__ = '1.0.0-beta'

from pathlib import Path

import arviz as az  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore
from pkg_resources import resource_filename  # type: ignore

import stan  # type: ignore

STAN_FILES_FOLDER = Path(__file__).parent / "stan"


def _generate_random_seed():
    return int(np.random.default_rng().integers(low=0,
                                                high=2**31,
                                                endpoint=False))


def _compile_stan_model(fname: str,
                        data: dict,
                        random_seed=None) -> stan.model.Model:
    """
    Compile the Stan model with the given file name using the given data.
    """
    with open(STAN_FILES_FOLDER / fname) as f:
        program_code = f.read()

    if random_seed is None:
        random_seed = _generate_random_seed()

    model: stan.model.Model = stan.build(program_code,
                                         data=data,
                                         random_seed=random_seed)

    return model


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

        n_instances, n_algorithms = self.metrics.shape

        self.data = dict(
            n_instances=n_instances,
            n_algorithms=n_algorithms,
            orders=orders,
            weights_instances=np.ones(n_instances),
            # Uniform prior.
            alpha=np.ones(n_algorithms),
        )

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
        fname = "pl_model.stan"

        self.model_: stan.model.Model = _compile_stan_model(
            fname, data=self.data, random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        self.infdata_: arviz.InferenceData = az.from_pystan(
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

        summary = az.summary(self.infdata_)
        print(summary)
        az.plot_forest(self.infdata_, var_names=["~loglik", "~rest"])
        az.plot_posterior(self.infdata_)
        az.plot_trace(self.infdata_)
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
    This model assumes that the data points for each algorithm are be i.i.d.
    This entails that the model does *not* take into account the correlation
    induced by cross-validation or similar methods.
    """

    def __init__(self, y1, y2):
        """
        Parameters
        ----------
        y1 : array of shape (n_tasks1,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the first method to compare.
        y2 : array of shape (n_tasks2,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the second method to compare.

        Notes
        -----
        As of now, this expects arrays (in particular, `pandas.DataFrame` or
        `pandas.Series` are not supported as inputs—use their `to_numpy()`
        method before passing them here).

        The arrays may have differing lengths as the model does not assume the
        samples to be paired.
        """
        self.y1 = y1
        self.y2 = y2

        n_runs1, = self.y1.shape
        n_runs2, = self.y2.shape

        self.data = dict(n_runs1=n_runs1,
                         n_runs2=n_runs2,
                         y1=self.y1,
                         y2=self.y2)

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
        fname = "kruschke.stan"

        self.model_: stan.model.Model = _compile_stan_model(
            fname, data=self.data, random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        self.infdata_: arviz.InferenceData = az.from_pystan(
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
        summary = az.summary(self.infdata_)
        print(summary)

        az.plot_ppc(self.infdata_,
                    kind="kde",
                    data_pairs={
                        "y1": "y1_rep",
                        "y2": "y2_rep"
                    },
                    num_pp_samples=10)
        az.plot_posterior(self.infdata_)
        az.plot_trace(self.infdata_)
        az.plot_density(self.infdata_.posterior.mu2
                        - self.infdata_.posterior.mu1,
                        hdi_markers="v")
        plt.show()


class BimodalNonNegative:
    """
    A Bayesian model that can be used to make statistical statements about the
    difference between two algorithms when run multiple times *independently* on
    a task and the units are

    - distributed bimodally
    - non-negative

    The model uses a simple mixture consisting of two Gamma distributions. For
    the exact specifications (e.g. priors etc.), see the Stan file.

    Originally created to model and compare the running times of algorithms
    where a small subset of the runs was considerably faster than the large
    majority.

    Notes
    -----
    This model assumes that the data points for each algorithm are be i.i.d.
    This entails that the model does *not* take into account the correlation
    induced by cross-validation or similar methods.
    """

    def __init__(self, y1, y2):
        """
        Parameters
        ----------
        y1 : array of shape (n_tasks1,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the first method to compare.
        y2 : array of shape (n_tasks2,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the second method to compare.

        Notes
        -----
        As of now, this expects arrays (in particular, `pandas.DataFrame` or
        `pandas.Series` are not supported as inputs—use their `to_numpy()`
        method before passing them here).

        The arrays may have differing lengths as the model does not assume the
        samples to be paired.
        """
        self.y1 = y1
        self.y2 = y2

        n_runs1, = self.y1.shape
        n_runs2, = self.y2.shape

        self.data = dict(
            n_runs1=n_runs1,
            n_runs2=n_runs2,
            y1=self.y1,
            y2=self.y2,
            # Assume the variances of the submodels to lie within [var_lower *
            # Var(y), var_upper * Var(y)] for y from {y1, y2}.
            var_lower=0.001,
            var_upper=1.0,
        )

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
        fname = "bimodal_nonnegative.stan"

        self.model_: stan.model.Model = _compile_stan_model(
            fname, data=self.data, random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        # TODO Fill InferenceData fully
        self.infdata_: arviz.InferenceData = az.from_pystan(
            posterior=self.fit_,
            posterior_predictive=["y1_rep", "y2_rep"],
            # predictions
            # prior
            # prior_predictive
            observed_data=["y1", "y2"],
            # constant_data
            # predictions_constant_data
            # log_likelihood
            # TODO coords
            # TODO dims
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
        summary = az.summary(self.infdata_, filter_vars="like")
        print(summary)

        post = self.infdata_.posterior

        # You could sample PPC yourself like this:
        # post_pred = self.infdata_.posterior_predictive
        # for i in range(10):
        #     chain = np.random.choice(post_pred.chain)
        #     draw = np.random.choice(post_pred.draw)
        #     ax[0].scatter(post_pred.y1_rep[chain, draw],
        #                   st.norm(loc=i + 1, scale=0.02).rvs(
        #                       len(post_pred.y1_rep[chain, draw])),
        #                   marker="+")

        plt.style.use('tableau-colorblind10')

        fig, ax = plt.subplots(2, layout="constrained", figsize=(10, 10))

        ax[0].hist(self.y1,
                   bins=100,
                   density=True,
                   label="Data",
                   alpha=0.5,
                   color="C0")
        ax[1].hist(self.y2,
                   bins=100,
                   density=True,
                   label="Data",
                   alpha=0.5,
                   color="C0")

        az.plot_ppc(self.infdata_,
                    observed=False,
                    colors=["C1", "C2", "C3"],
                    data_pairs={
                        "y1": "y1_rep",
                        "y2": "y2_rep"
                    },
                    num_pp_samples=10,
                    ax=ax)

        # Compute means over these dims.
        dim = ["chain", "draw"]

        weight1 = post.weight1.mean(dim=dim).to_numpy()
        alpha1 = post.alpha1.mean(dim=dim).to_numpy()
        beta1 = post.beta1.mean(dim=dim).to_numpy()
        dist1 = st.gamma(alpha1, scale=1 / beta1)

        X1 = np.linspace(np.min(dist1.ppf(0.01)), np.max(dist1.ppf(0.99)),
                         1000)

        y1 = np.sum(dist1.pdf(np.repeat(X1[:, np.newaxis], 2, axis=1))
                    * np.array([weight1, 1 - weight1]),
                    axis=1)
        ax[0].plot(X1,
                   y1,
                   label="Posterior density based on parameter means",
                   color="C4")
        ax[0].vlines(alpha1 / beta1,
                     0,
                     ax[0].get_ylim()[1],
                     linestyle="dotted",
                     color="C5",
                     label="Posterior component means")
        ax[0].legend()

        weight2 = post.weight2.mean(dim=dim).to_numpy()
        alpha2 = post.alpha2.mean(dim=dim).to_numpy()
        beta2 = post.beta2.mean(dim=dim).to_numpy()
        dist2 = st.gamma(alpha2, scale=1 / beta2)

        X2 = np.linspace(np.min(dist2.ppf(0.01)), np.max(dist2.ppf(0.99)),
                         1000)

        y2 = np.sum(dist2.pdf(np.repeat(X2[:, np.newaxis], 2, axis=1))
                    * np.array([weight2, 1 - weight2]),
                    axis=1)
        ax[1].plot(X2,
                   y2,
                   label="Posterior density based on parameter means",
                   color="C4")
        ax[1].vlines(alpha2 / beta2,
                     0,
                     ax[1].get_ylim()[1],
                     linestyle="dotted",
                     color="C5",
                     label="Posterior component means")

        az.plot_posterior(self.infdata_)
        az.plot_trace(self.infdata_)
        plt.show()


class NonNegative:
    """
    A Bayesian model that can be used to make statistical statements about the
    difference between two algorithms when run multiple times *independently* on
    a task and the units are

    - distributed unimodally
    - non-negative
    - optional: cut off at some point (e.g. if runs have been aborted at a
      maximum number of steps; also known as *right-censored data*)

    The model uses a simple Gamma distribution for each data set. For the exact
    specifications (e.g. priors etc.), see the Stan file.

    Originally created to model and compare the running times of algorithms.

    Notes
    -----
    This model assumes that the data points for each algorithm are be i.i.d.
    This entails that the model does *not* take into account the correlation
    induced by cross-validation or similar methods.
    """

    def __init__(self,
                 y1,
                 y2,
                 var_lower=None,
                 var_upper=None,
                 mean_upper=None,
                 n_censored1=0,
                 n_censored2=0,
                 censoring_point=3000.0):
        """
        Parameters
        ----------
        y1 : array of shape (n_tasks1,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the first method to compare.
        y2 : array of shape (n_tasks2,)
            Independently (i.e. no cross-validation etc.) generated performance
            statistics values (e.g. MSE, accuracy, maximum fitness, average RL
            return, …) of the second method to compare.
        var_lower, var_upper : float > 0 or None
            Hyperprior parameters on the variances. Assume the variances of the
            submodels to lie within [var_lower * Var(y), var_upper * Var(y)] for
            y from {y1, y2}. If `None`, use the default non-committal values of
            `0.001` and `1000.0` which are the ones used by Kruschke in his 2013
            paper.
        mean_upper : float > mean(y)/100
            Hyperprior parameter on the means. We assume that with a probability
            of 90%, the real data means lie in [min(y)/100, min(y)/100 +
            mean_upper]. If `None`, use the default of `2 * max(y)`.
        n_censored1, n_censored2 : int >= 0
            Number of censored data points for each of the methods.
        censoring_point : float > 0
            Value above which data has been censored. Only relevant if at least
            one of `n_censored1` and `n_censored2` is greater than 0.

        Notes
        -----
        As of now, this expects arrays (in particular, `pandas.DataFrame` or
        `pandas.Series` are not supported as inputs—use their `to_numpy()`
        method before passing them here).

        The arrays may have differing lengths as the model does not assume the
        samples to be paired.
        """
        self.y1 = y1
        self.y2 = y2

        self.var_lower = var_lower if var_lower is not None else 0.001
        self.var_upper = var_upper if var_upper is not None else 1000.0
        # TODO Consider to support setting this for each method individually
        self.mean_upper = (mean_upper if mean_upper is not None else 2.0
                           * y2.max())

        self.n_censored1 = n_censored1
        self.n_censored2 = n_censored2
        self.censoring_point = censoring_point

        n_runs1, = self.y1.shape
        n_runs2, = self.y2.shape

        self.data = dict(
            n_runs1=n_runs1,
            n_runs2=n_runs2,
            y1=self.y1,
            y2=self.y2,
            var_lower=self.var_lower,
            var_upper=self.var_upper,
            mean_upper=self.mean_upper,
            n_censored1=self.n_censored1,
            n_censored2=self.n_censored2,
            censoring_point=self.censoring_point,
        )

    def fit(self, random_seed=None, **kwargs):
        """
        Compares the two samples using this model.

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
        fname = "nonnegative.stan"

        self.model_: stan.model.Model = _compile_stan_model(
            fname, data=self.data, random_seed=random_seed)

        self.fit_: stan.fit.Fit = self.model_.sample(**kwargs)

        # TODO Fill InferenceData fully
        self.infdata_: arviz.InferenceData = az.from_pystan(
            posterior=self.fit_,
            posterior_predictive=["y1_rep", "y2_rep"],
            # predictions
            # prior
            # prior_predictive
            observed_data={
                "y1": self.data["y1"],
                "y2": self.data["y2"]
            },
            # constant_data
            # predictions_constant_data
            # log_likelihood
            # coords
            # dims
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
        summary = az.summary(self.infdata_, filter_vars="like")
        print(summary)

        summary = az.summary(self.infdata_)
        print(summary)

        az.plot_ppc(self.infdata_,
                    kind="kde",
                    data_pairs={
                        "y1": "y1_rep",
                        "y2": "y2_rep"
                    },
                    num_pp_samples=10)
        az.plot_posterior(self.infdata_)
        az.plot_trace(self.infdata_)
        az.plot_density(self.infdata_.posterior.mean2
                        - self.infdata_.posterior.mean1,
                        hdi_markers="v")
        plt.show()


class BayesCorrTTest:
    """
    A Bayesian model that can be used to make statistical statements about the
    difference between two algorithms when run multiple times on a task using
    cross validation.

    The underlying model is introduced in the 2015 article by Corani and
    Benavoli, *A Bayesian approach for comparing cross-validated algorithms on
    multiple data sets* (in that publication it's called the *Bayesian t test
    for correlated observations model* equation (8)).

    Notes
    -----
    This model assumes that the data points for each algorithm are be i.i.d.
    This entails that the model does *not* take into account the correlation
    induced by cross-validation or similar methods.

    This model does not use MCMC/Stan since the posterior is analytically
    tractable. After fitting, its `model_` attribute is a `scipy.stats.t` object
    which can be queried further. Note that, nevertheless, `infdata_` is an
    `arviz.InferenceData` containing a sample of the specified size for a more
    unified interface.
    """

    # TODO What about the case where the algorithm is run multiple times *and*
    # CV is used?

    def __init__(self, y1, y2, fraction_test):
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
        fraction_test : float
            The fraction of the data used as test set (i.e. `n_test / (n_test +
            n_train)` used to estimate the correlation introduced by CV, based
            on (Nadeau and Bengio, 2003)).

        Notes
        -----
        As of now, this expects arrays (in particular, `pandas.DataFrame` or
        `pandas.Series` are not supported as inputs—use their `to_numpy()`
        method before passing them here).
        """
        self.y1 = y1
        self.y2 = y2
        self.fraction_test = fraction_test

    def fit(self, random_seed=None, num_samples=10000):
        """
        Compares the two samples using the *Bayesian t test for correlated
        observations model* described in the 2015 article by Corani and
        Benavoli, *A Bayesian approach for comparing cross-validated algorithms
        on multiple data sets*.

        Parameters
        ----------
        random_seed : non-negative int < 2**31 - 1
            Random seed to be used for sampling. See
            [`stan.build`](https://pystan.readthedocs.io/en/latest/reference.html).
        num_samples : int
            Number of samples to draw from the distribution.
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
        x = self.y2 - self.y1

        x_over = x.mean()
        n = len(x)
        sigma_2_hat = ((x - x_over)**2).sum() / (n - 1)
        rho = self.fraction_test

        # Equation (8) from Corani and Benavoli, 2015, *A Bayesian approach for
        # comparing cross-validated algorithms on multiple data sets*.
        self.model_ = st.t(df=n - 1,
                           loc=x_over,
                           scale=(1 / n + rho / (1 - rho)) * sigma_2_hat)
        # TODO Consider splitting num_samples into chains to prevent arviz warning
        self.fit_: np.ndarray = self.model_.rvs(num_samples)
        self.infdata_: arviz.InferenceData = az.convert_to_inference_data(
            self.fit_)

        return self

    def _analyse(self):
        summary = az.summary(self.infdata_)
        print(summary)
        az.plot_posterior(self.infdata_,
                          rope={"posterior": {
                              "rope": (-0.01, 0.01)
                          }})
        plt.show()
