#ifndef REFILL_UTILITY_RESAMPLE_METHODS_H_
#define REFILL_UTILITY_RESAMPLE_METHODS_H_

#include <glog/logging.h>

#include <random>

namespace refill {

class SamplingFunctorBase {
 public:
  virtual ~SamplingFunctorBase() = default;

  virtual void operator()(Eigen::MatrixXd* particles,
                          Eigen::VectorXd* weights) {
  }
};

class ImportanceSamplingFunctor : public SamplingFunctorBase {
 public:
  virtual ~ImportanceSamplingFunctor() = default;

  virtual void operator()(Eigen::MatrixXd* particles,
                          Eigen::VectorXd* weights) {
    const std::size_t n_particles = particles->cols();

    CHECK_NE(n_particles, 0)<< "Particle vector is empty.";

    // Compute the cumulative sum of weights
    Eigen::VectorXd cum_sum(n_particles);

    cum_sum(0) = (*weights)(0);
    for (int i = 1; i < n_particles; ++i) {
      cum_sum(i) = cum_sum(i - 1) + (*weights)(i);
    }

    // Set up rng and uniform distribution
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    // Create new vector of resampled particles
    Eigen::MatrixXd particles_copy(*particles);
    for (int i = 0; i < n_particles; ++i) {
      double particle_random_num = uniform_dist(rng_);

      for (int j = 0; j < n_particles; ++j) {
        if (particle_random_num < cum_sum[j]) {
          particles->col(i) = particles_copy.col(j);
          break;
        }
      }
    }

    weights->setConstant(1 / n_particles);
  }

  /** @brief Random number generator used for random sampling. */
  std::mt19937 rng_;
};

class ThresholdedImportanceSamplingFunctor : public SamplingFunctorBase {
 public:
  explicit ThresholdedImportanceSamplingFunctor(
      double effective_particle_number)
      : effective_particle_num_(effective_particle_number) {
  }

  virtual void operator()(Eigen::MatrixXd* particles,
                          Eigen::VectorXd* weights) {
    double squared_norm = weights->squaredNorm();
    CHECK_NE(squared_norm, 0.0)<< "Weight squared norm is zero!";

    double N_eff = 1 / squared_norm;

    // Only resample if the effective number of particles is lower
    // than the threshold
    if (N_eff < effective_particle_num_) {
      sampling_object_(particles, weights);
    }
  }

  ImportanceSamplingFunctor sampling_object_;
  double effective_particle_num_;
};

}  // namespace refill

#endif  // REFILL_UTILITY_RESAMPLE_METHODS_H_
