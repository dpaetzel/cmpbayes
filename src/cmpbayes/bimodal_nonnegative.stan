/* A mixture model consisting of two Gamma distributions.
 *
 * Priors on the parameters of the Gamma distributions are based on the means of
 * the Gamma distributions being assumed to stem from an exponential
 * distribution while the variances lie in an interval derived from the variance
 * in the data.
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
  // gamma distributions.
  real<lower=0, upper=1> var_lower;
  real<lower=var_lower, upper=1> var_upper;
}


transformed data {
  int n_modes = 2;

  // We want 90% of (exponentially distributed) means to lie in [0, max(y)].
  real lambda_mean1 = - log(1 - 0.9) / max(y1);
  real lambda_mean2 = - log(1 - 0.9) / max(y2);

  real mean_weight = 0.5;
  real var_weight = square(0.1);

  real alpha_weight =
    square(mean_weight) / var_weight
       - pow(mean_weight, 3) / var_weight - mean_weight;
  real beta_weight = alpha_weight / mean_weight - alpha_weight;

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
  real<lower=0, upper=1> weight1;
  real<lower=0, upper=1> weight2;

  positive_ordered[n_modes] mean1_unshifted;
  positive_ordered[n_modes] mean2_unshifted;

  vector
  <lower=(mean1_unshifted + min_mean1) / var_y1_upper,
    upper=(mean1_unshifted + min_mean1) / var_y1_lower>
    [n_modes]
    beta1;

  vector
  <lower=(mean2_unshifted + min_mean2) / var_y2_upper,
    upper=(mean2_unshifted + min_mean2) / var_y2_lower>
    [n_modes]
    beta2;
}


transformed parameters {
  vector<lower=min_mean1>[n_modes] mean1 = mean1_unshifted + min_mean1;
  vector<lower=min_mean2>[n_modes] mean2 = mean2_unshifted + min_mean2;

  // .* is elementwise product.
  vector<lower=0>[n_modes] alpha1 = mean1 .* beta1;
  vector<lower=0>[n_modes] alpha2 = mean2 .* beta2;
}


model {
  // Note that this very seldomly yields a mean1_unshifted of inf which then
  // leads to an exception and rejection of the current sample.
  mean1_unshifted ~ exponential(lambda_mean1);
  mean2_unshifted ~ exponential(lambda_mean2);

  weight1 ~ beta(alpha_weight, beta_weight);
  weight2 ~ beta(alpha_weight, beta_weight);

  beta1 ~ uniform(mean1 / var_y1_upper, mean1 / var_y1_lower);
  beta2 ~ uniform(mean2 / var_y2_upper, mean2 / var_y2_lower);

  for (n in 1:n_runs1) {
    target +=
      log_mix(weight1,
              gamma_lpdf(y1[n] | alpha1[1], beta1[1]),
              gamma_lpdf(y1[n] | alpha1[2], beta1[2]));
  }
  for (n in 1:n_runs2) {
    target +=
      log_mix(weight2,
              gamma_lpdf(y2[n] | alpha2[1], beta2[1]),
              gamma_lpdf(y2[n] | alpha2[2], beta2[2]));
  }
}


generated quantities {
  // TODO Add prior predictive check.

  // Posterior predictive check.
  array [n_runs1] real y1_rep;
  for (i in 1:n_runs1) {
    real z = bernoulli_rng(weight1);
    y1_rep[i] =
      z * gamma_rng(alpha1[1], beta1[1])
      + (1 - z) * gamma_rng(alpha1[2], beta1[2]);
  }
  array [n_runs2] real y2_rep;
  for (i in 1:n_runs2) {
    real z = bernoulli_rng(weight2);
    y2_rep[i] =
      z * gamma_rng(alpha2[1], beta2[1])
      + (1 - z) * gamma_rng(alpha2[2], beta2[2]);
  }
}
