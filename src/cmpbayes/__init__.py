__version__ = '0.0.1'

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from pkg_resources import resource_filename

pl_stan_filename = resource_filename(__name__, "pl_model.stan")


def fit_calvo(metrics, higher_better=True, **kwargs):
    """
    Builds a Bayesian model of the ranking of the algorithms that produced the
    provided data.

    The underlying model is the described in the 2018 article by Calvo et al.,
    Bayesian Inference for Algorithm Ranking Analysis.

    Parameters
    ----------
    metrics : array of shape (n_instances, n_algorithms)
        Each row corresponds to one problem instance (e.g. one optimization
        problem, one learning task, …). Columns correspond to algorithms.
        Entries are performance statistics values (e.g. MSE, accuracy, maximum
        fitness, average RL return, …).
    higher_better: bool
        Whether higher metrics values are better than lower ones. This is e.g.
        the case for accuracy on classification tasks but not for MSE on
        regression tasks. This is required in order to be able to correctly
        compute the algorithm ranking.
    kwargs : kwargs
        Are passed through to ``stan.model.Model.sample``. You may set
        ``num_samples``, ``num_warmup`` and many more options here. See the
        documentation of `sample
        <https://pystan.readthedocs.io/en/latest/reference.html#stan.model.Model.sample>`_
        as well as the `code of the sampler currently used
        <https://github.com/stan-dev/stan/blob/develop/src/stan/services/sample/hmc_nuts_diag_e_adapt.hpp>`_.

    Returns
    -------
    object
        A `stan.fit.Fit object
        <https://pystan.readthedocs.io/en/latest/reference.html#stan.fit.Fit>`_.
    """
    # For each instance, get the sorting permutation of the algorithms (i.e. the
    # indexes of the algorithms in the order such that their metrics are sorted
    # ascendingly).
    orders = np.argsort(metrics, axis=1)

    # np.argsort sorts ascendingly, if higher is better we thus want to reverse the
    # ordering such that the best stands at the front.
    if higher_better:
        np.flip(orders, axis=1)

    # Stan starts indexes from 1 and each element of orders is an index (i.e. an
    # algorithm number).
    orders = orders + 1

    # Read the Stan model from disk.
    with open(pl_stan_filename) as f:
        program_code = f.read()

    data = {
        "n_instances": n_instances,
        "n_algorithms": n_algorithms,
        "orders": orders,
        "weights_instances": np.ones(n_instances),
        # Uniform prior.
        "alpha": np.ones(n_algorithms),
    }

    model: stan.model.Model = stan.build(program_code, data=data)

    fit: stan.fit.Fit = model.sample(**kwargs)

    return fit


if __name__ == '__main__':
    seed = 2
    rng = np.random.default_rng(seed)

    n_instances = 1000

    algorithm_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    n_algorithms = len(algorithm_names)

    metrics = rng.normal(loc=100, scale=40, size=(n_instances, n_algorithms))

    fit = fit_calvo(metrics, higher_better=True)

    df = fit.to_frame()

    weights = df.filter(regex="^weights.*$")
    weights.columns = pd.Index(algorithm_names, name="p(_ ranked first)")

    print(weights.mean())
    print(weights.quantile([0.05, 0.95]))

    # TODO Add ppd example, see
    # https://mc-stan.org/docs/2_24/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html

    # See https://oriolabril.github.io/arviz/api/generated/arviz.from_pystan.html .
    azd = az.from_pystan(
        posterior=fit,
        # posterior_predictive="weights_hat",
        observed_data=["orders"],
        log_likelihood={"weights": "loglik"},
        # coords=…,
        # dims=…,
    )

    az.plot_posterior(azd)
    az.plot_trace(azd)
    plt.show()
