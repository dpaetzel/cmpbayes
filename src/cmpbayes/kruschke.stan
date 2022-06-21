/* The model described in the 2013 article by Kruschke, *Bayesian Estimation
 * Supersedes the t Test*.
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
}


transformed data {
  // The “pooled data”.
  vector[n_runs1 + n_runs2] y = append_row(y1, y2);

  // “To keep the prior distribution broad relative to the arbitrary scale of
  // the data, I have set the standard deviation S of the prior on mu to 1,000
  // times the standard deviation of the pooled data.”
  real S = 1000 * sd(y);

  // “The mean M of the prior on mu is arbitrarily set to the mean of the pooled
  // data; this setting is done merely to keep the prior scaled appropriately
  // relative to the arbitrary scale of the data.”
  real M = mean(y);

  // “The prior on the standard deviation parameter is also assumed to be
  // non-committal, expressed as a uniform distribution from a low value L, set
  // to one thousandth of the standard deviation of the pooled data, to a high
  // value H, set to one thousand times the standard deviation of the pooled
  // data.”
  real L = 1.0 / 1000 * sd(y);
  real H = 1000 * sd(y);

  // Hyperparameter for the prior on nu.
  real lambda = 1.0 / 29;
}

parameters {
  // Means of the metrics.
  real mu1;
  real mu2;

  // Standard deviations of the metrics.
  real<lower=L,upper=H> sigma1;
  real<lower=L,upper=H> sigma2;

  // Normality parameter of the Student's t distributions that are fitted to the
  // data.
  real<lower=0> nu_minus_one;
}

transformed parameters {
  real<lower=1> nu = nu_minus_one + 1;
}

model {
  mu1 ~ normal(M, S);
  mu2 ~ normal(M, S);

  sigma1 ~ uniform(L, H);
  sigma2 ~ uniform(L, H);

  nu_minus_one ~ exponential(lambda);

  y1 ~ student_t(nu, mu1, sigma1);
  y2 ~ student_t(nu, mu2, sigma2);
}

generated quantities {
  // What we actually want to find out most of the time.
  real mu2_minus_mu1 = mu2 - mu1;

  // Kruschke also plots this.
  real sigma2_minus_sigma1 = sigma2 - sigma1;

  // For posterior predictive checking.
  array [n_runs1] real y1_rep =
    student_t_rng(nu, rep_array(mu1, n_runs1), sigma1);
  array [n_runs2] real y2_rep =
    student_t_rng(nu, rep_array(mu2, n_runs2), sigma2);
}
