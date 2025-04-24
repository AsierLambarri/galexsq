data {
    int<lower=0> N;
    int<lower=0> M;
    vector[N] x;
    vector[N] y;
    vector[M] xeval;
}

parameters {
    real beta0;
    real beta1;
    real<lower=0> sigma;
    real<lower=0> nu;
}

model {
    y ~ student_t(nu, beta0 + beta1 * x, sigma);
    beta0 ~ normal(0., 100.);
    beta1 ~ normal(0., 100.);
    sigma ~ cauchy(0., 100.);
    nu ~ exponential(1./30.);
}

generated quantities {
    vector[N] logLikelihood;
    vector[M] dpp;
        
    for (i in 1:N){
      logLikelihood[i] = student_t_lpdf(x[i] | nu, beta0 + beta1 * x[i], sigma ) ;
    }
    for (i in 1:M){
      dpp[i] = student_t_rng(nu, beta0 + beta1 * xeval[i], sigma) ;
    }
    
    
}