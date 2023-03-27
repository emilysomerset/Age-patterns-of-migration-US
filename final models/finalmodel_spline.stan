/* RW on overdispersion */
data {
  int<lower=0> A; //number of age groups
  int<lower=0> R; //number of regions
  int<lower=0> N; //number of data points
  int<lower=0> T; //number of years 
  int<lower=0> K; //number of knots
  int<lower=0> y[A*R*T]; //vector of migrants
  vector[A*R*T] pop; //vector of log population
  matrix[N,(A-1)] age; //matrix of ages 
  matrix[N,(R-1)] region; //matrix of regions
  matrix[N,(T-1)] time; //matrix of years
  matrix[N,(A-1)*(R-1)] age_region; //interaction
  int<lower=1> age_vec[A*R*T];
  int<lower=1> time_vec[A*R*T]; 
  int<lower=1> region_vec[A*R*T]; 
  matrix[A*R,(A-1)] pred_age; //matrix of ages 
  matrix[A*R,(R-1)] pred_region; //matrix of regions
  matrix[A*R,(A-1)*(R-1)] pred_age_region; //interaction
  int<lower=1> pred_age_vec[A*R];
  int<lower=1> pred_region_vec[A*R];
  matrix[T,K] B; 
}
parameters {
  vector[A-1] tau; //age-specific log rates
  vector[R-1] gamma; //region-specific intercept for log rates
  // vector[(A-1)*(R-1)] xi;
  real alpha;// intercept
  matrix[A,R] delta1; //matrix of deviations
  matrix[T,R] delta2; //matrix of deviations
  // matrix[A,R] delta3; //matrix of deviations
  real<lower=0> sigma1; //sd of delta1
  real<lower=0> sigma2[R]; //sd of delta2
}

transformed parameters{
  vector[N] log_rate; //vector of age & region specific log rate
  vector[N] fixed_eff;
  vector[N] deviation_fixed_eff;
  fixed_eff = alpha + age*tau + region*gamma;
  for (i in 1:N){
      deviation_fixed_eff[i] = fixed_eff[i] + delta1[age_vec[i],region_vec[i]];
      log_rate[i] = fixed_eff[i] + delta1[age_vec[i],region_vec[i]] + delta2[time_vec[i],region_vec[i]];
      }
    }
  

model {
  //likelihood
      y ~ poisson_log(log_rate + pop);


  //priors
  tau ~ normal(0,1);
  alpha ~ normal(0,1);
  gamma ~ normal(0,1);
  

// Region and age delta
//----------------------------------------

sigma1~ normal(0,1);

for (a in 1:A){
  sum(delta1[a, 1:R])~ normal(0, 0.001);
}

delta1[1,1:R] ~ normal(0, sigma1);
for (a in 2:A){
  delta1[a, 1:R] ~ normal(delta1[(a-1),1:R], sigma1);
}
//----------------------------------------

  
// Region and time delta
//-----------------------------------------------
sigma2~ normal(0,2);

delta2[1,1:R] ~ normal(0, sigma2);
for (t in 2:T){
    delta2[t, 1:R] ~ normal(delta2[(t-1), 1:R], sigma2);
}

//----------------------------------------

}
  
  
generated quantities {

vector[N] rate_p;
int<lower=0> y_p[N];
vector[N] generated_rate;
vector[A*R] pred_rate;
vector[A*R] pred_fixed_eff;

y_p = poisson_rng(exp(log_rate + pop));

for (i in 1:N){
generated_rate[i] = y_p[i]/exp(pop[i]);
if (time_vec[i]==1){rate_p[i] = 0;
} else{
  rate_p[i] = fixed_eff[i] + delta1[age_vec[i],region_vec[i]]+ normal_rng(delta2[(time_vec[i]-1),region_vec[i]],sigma2[region_vec[i]]);
}}

pred_fixed_eff = alpha + pred_age*tau + pred_region*gamma;
// pred_age_region*xi;
for (i in 1:(A*R)){
 pred_rate[i] = pred_fixed_eff[i]+ delta1[pred_age_vec[i],pred_region_vec[i]] + normal_rng(delta2[T,pred_region_vec[i]],sigma2[region_vec[i]]);
 }
}

