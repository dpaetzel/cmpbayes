// Implementation of the model from the 2020 article by Calvo et al.: Bayesian
// performance analysis for black-box optimization benchmarking .
//
// Based on the authors' code which can be found at
// https://github.com/b0rxa/scmamp/blob/master/inst/stan/pl_model.stan .
data {
  // Number of instances.
  int<lower=1> n_instances;
  // Number of algorithms.
  int<lower=2> n_algorithms;

  // Matrix with all the orders, one per row. order[1, 2] == 3 means that
  // algorithm 3 is 2nd on instance 1. In the accompanying text, this is
  // called R and consists of n_instances many sigmas.
  array[n_instances,n_algorithms] int orders;

  // Parameters for the Dirichlet prior.
  vector[n_algorithms] alpha;

  // Weights for the instances (optional, probably just set this to an all
  // ones vector).
  array[n_instances] real weights_instances;
}


parameters {
  // The simplex constrains the weights to sum to 1.
  simplex[n_algorithms] weights;
}

transformed parameters{
  real loglik;
  real rest;

  loglik=0;
  for (s in 1:n_instances){
    // We can cut the last one because the next loop won't do anything then.
    for (i in 1:(n_algorithms-1)){
      rest=0;
      for (j in i:n_algorithms){
        rest = rest + weights[orders[s, j]];
      }
      // We allow to assign weights to the instances here.
      loglik = loglik + log(weights_instances[s] * weights[orders[s, i]] / rest);
    }
  }
}

model {
  weights ~ dirichlet(alpha);
  target += loglik;
}

// A PPC would be good but I'm not sure how to plot one for this particular
// model.
// https://mc-stan.org/docs/2_24/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html
