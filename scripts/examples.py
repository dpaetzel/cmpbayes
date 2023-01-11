import click  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore


@click.group()
def cli():
    pass


@cli.command()
def bayescorrttest():
    """
    Run test scenario for the Bayes correlated t test model.
    """
    from cmpbayes import BayesCorrTTest

    seed = 4
    rng = np.random.default_rng(seed)

    n_runs = 30

    y1 = rng.normal(loc=100, scale=20, size=(n_runs))
    y2 = rng.normal(loc=103, scale=10, size=(n_runs))

    model = BayesCorrTTest(y1, y2,
                           fraction_test=0.25).fit(random_seed=seed + 1)

    model._analyse()


@cli.command()
def calvo():
    """
    Run test scenario for the Calvo model.
    """
    from cmpbayes import Calvo

    seed = 2
    rng = np.random.default_rng(seed)

    n_instances = 1000

    algorithm_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
    n_algorithms = len(algorithm_labels)

    metrics = rng.normal(loc=100, scale=40, size=(n_instances, n_algorithms))

    model = Calvo(metrics,
                  higher_better=True,
                  algorithm_labels=algorithm_labels).fit(random_seed=seed + 1)
    model._analyse()


@cli.command()
def kruschke():
    """
    Run test scenario for the Kruschke model.
    """
    from cmpbayes import Kruschke

    seed = 3
    rng = np.random.default_rng(seed)

    n_runs = 30

    y1 = rng.normal(loc=100, scale=20, size=(n_runs))
    y2 = rng.normal(loc=103, scale=10, size=(n_runs))

    model = Kruschke(y1, y2).fit(random_seed=seed + 1)
    model._analyse()


@cli.command()
def bimodnneg():
    """
    Run test scenario for the non-negative bimodal model.
    """
    from cmpbayes import BimodalNonNegative

    from _timedata import y1, y2

    iter_sampling = 5000
    seed = 1

    model = BimodalNonNegative(y1, y2).fit(num_samples=iter_sampling,
                                           random_seed=seed)
    model._analyse()


@cli.command()
@click.option("--var-lower", default=None, type=float)
@click.option("--var-upper", default=None, type=float)
@click.option("--mean-upper", default=None, type=float)
def nneg(var_lower, var_upper, mean_upper):
    """
    Run test scenario for the non-negative unimodal model.
    """
    from cmpbayes import NonNegative

    # scale == 1 / beta.
    size = 20
    y1 = st.gamma.rvs(a=3, scale=1 / 10, size=size)
    y2 = st.gamma.rvs(a=4, scale=1 / 10, size=size)

    num_samples = 5000
    seed = 1

    n_censored1 = 2
    n_censored2 = 5
    censoring_point = 10

    model = NonNegative(y1,
                        y2,
                        var_lower=var_lower,
                        var_upper=var_upper,
                        mean_upper=mean_upper,
                        n_censored1=n_censored1,
                        n_censored2=n_censored2,
                        censoring_point=censoring_point).fit(
                            num_samples=num_samples, random_seed=seed)

    model._analyse()

    post = model.infdata_.posterior
    diff = post.mean2.to_numpy() - post.mean1.to_numpy()
    plt.hist(diff.ravel(),
             bins=100,
             density=True,
             label=(f"n_censored1={n_censored1}, "
                    f"n_censored2={n_censored2}, "
                    f"censoring_point={censoring_point}"))
    plt.show()


@cli.command()
def betabin():
    """
    Run test scenario for the beta-binomial model.
    """
    from cmpbayes import BetaBinomial

    n_success1 = 2
    n_success2 = 3
    n1 = 20
    n2 = 21

    model = BetaBinomial(n_success1 = n_success1,
                         n_success2 = n_success2,
                         n1 = n1,
                         n2 = n2).fit()
    model._analyse()


if __name__ == '__main__':
    cli()
