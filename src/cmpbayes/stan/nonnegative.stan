/* A model for comparing two sets of measurements of non-negative metrics.
 *
 * Each data set is modelled using a single Gamma distribution. Priors on the
 * parameters of the Gamma distribution are based on the mean of the Gamma
 * distributions being assumed to stem from an exponential distribution while
 * the variance lies in an interval derived from the variance in the data.
 */
data {
  // Number of runs in the first data set.
  int<lower=1> n_runs1;

  // Number of runs in the second data set.
  int<lower=1> n_runs2;

  // Units for the first method to be compared.
  vector[n_runs1] y1;

  // Units for the second method to be compared.
  vector[n_runs2] y2;

  // Factor for the lower and upper bound on the variance of the prior on the
  // gamma distribution.
  real<lower=0, upper=1> var_lower;
  real<lower=var_lower> var_upper;

  // Rate of unshifted means that should lie in [0, max(y)].
  real<lower=0, upper=1> mean_rate;
}


transformed data {
  // We want 90% of (exponentially distributed) unshifted means to lie in [0,
  // max(y)].
  real lambda_mean1 = - log(1 - mean_rate) / max(y1);
  real lambda_mean2 = - log(1 - mean_rate) / max(y2);

  // We cache these.
  real var_y1_upper = var_upper * variance(y1);
  real var_y1_lower = var_lower * variance(y1);
  real var_y2_upper = var_upper * variance(y2);
  real var_y2_lower = var_lower * variance(y2);

  // The minimum mean we'll consider for each of the components should be a
  // hundredth of the minimum of the data.
  real min_mean1 = min(y1) / 100;
  real min_mean2 = min(y2) / 100;
}


parameters {
  real<lower=0> mean1_unshifted;
  real<lower=0> mean2_unshifted;

  real
  <lower=(mean1_unshifted + min_mean1) / var_y1_upper,
    upper=(mean1_unshifted + min_mean1) / var_y1_lower>
    beta1;

  real
  <lower=(mean2_unshifted + min_mean2) / var_y2_upper,
    upper=(mean2_unshifted + min_mean2) / var_y2_lower>
    beta2;
}


transformed parameters {
  real<lower=min_mean1> mean1 = mean1_unshifted + min_mean1;
  real<lower=min_mean2> mean2 = mean2_unshifted + min_mean2;

  // .* is elementwise product.
  real<lower=0> alpha1 = mean1 .* beta1;
  real<lower=0> alpha2 = mean2 .* beta2;
}


model {
  // Note that this very seldomly yields a mean1_unshifted of inf which then
  // leads to an exception and rejection of the current sample.
  mean1_unshifted ~ exponential(lambda_mean1);
  mean2_unshifted ~ exponential(lambda_mean2);

  // weight1 ~ beta(alpha_weight, beta_weight);
  // weight2 ~ beta(alpha_weight, beta_weight);

  beta1 ~ uniform(mean1 / var_y1_upper, mean1 / var_y1_lower);
  beta2 ~ uniform(mean2 / var_y2_upper, mean2 / var_y2_lower);

  y1 ~ gamma(alpha1, beta1);
  y2 ~ gamma(alpha2, beta2);
}


generated quantities {
  // TODO Add prior predictive check.

  // What we actually want to find out most of the time.
  real mean2_minus_mean1 = mean2 - mean1;

  // Posterior predictive check.
  array [n_runs1] real y1_rep;
  for (i in 1:n_runs1) {
    y1_rep[i] = gamma_rng(alpha1, beta1);
  }
  array [n_runs2] real y2_rep;
  for (i in 1:n_runs1) {
    y2_rep[i] = gamma_rng(alpha2, beta2);
  }
}
