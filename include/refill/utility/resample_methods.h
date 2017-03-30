#ifndef REFILL_UTILITY_RESAMPLE_METHODS_H_
#define REFILL_UTILITY_RESAMPLE_METHODS_H_

#include <glog/logging.h>

#include <utility>
#include <random>
#include <vector>

namespace refill {

void importanceSampling(Eigen::MatrixXd* particles, Eigen::VectorXd* weights) {
  const std::size_t n_particles = particles->cols();

  CHECK_NE(n_particles, 0)<< "Particle vector is empty.";

  // Compute the cumulative sum of weights
  Eigen::VectorXd cum_sum(n_particles);

  cum_sum[0] = weights[0];
  for (int i = 1; i < n_particles; ++i) {
    cum_sum[i] = cum_sum[i - 1] + weights[i];
  }

  // Set up rng and uniform distribution
  std::random_device true_rng;
  std::default_random_engine random_engine(true_rng);
  std::uniform_real_distribution uniform_dist(0.0, 1.0);

  // Create new vector of resampled particles
  Eigen::MatrixXd particles_copy(*particles);
  for (int i = 0; i < n_particles; ++i) {
    double particle_random_num = uniform_dist(random_engine);

    for (int j = 0; j < n_particles; ++j) {
      if (particle_random_num < cum_sum[j]) {
        particles->col(i) = particles_copy.col(j);
        break;
      }
    }
  }

  weights->setConstant(1 / n_particles);
}

}  // namespace refill

#endif  // REFILL_UTILITY_RESAMPLE_METHODS_H_
