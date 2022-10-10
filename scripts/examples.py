import click # type: ignore
import numpy as np # type: ignore


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


if __name__ == '__main__':
    cli()
