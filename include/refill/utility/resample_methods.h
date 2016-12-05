#ifndef REFILL_UTILITY_RESAMPLE_METHODS_H_
#define REFILL_UTILITY_RESAMPLE_METHODS_H_

#include <glog/logging.h>

#include <utility>
#include <random>
#include <vector>

template<typename StateType>
using ParticleType = std::pair<StateType, double>;

namespace refill {

template<typename StateType>
void importanceSampling(std::vector<ParticleType<StateType>>* particles) {
  std::size_t n_particles = particles->size();

  CHECK_NE(n_particles, 0)<< "Particle vector is empty.";

  // Compute the cumulative sum of weights
  std::vector<double> cum_sum(n_particles);

  cum_sum[0] = particles->at(0).second;
  for (int i = 1; i < n_particles; ++i) {
    cum_sum[i] = cum_sum[i - 1] + particles->at(i).second;
  }

  // Set up rng and uniform distribution
  std::random_device true_rng;
  std::default_random_engine random_engine(true_rng);
  std::uniform_real_distribution uniform_dist(0.0, 1.0);

  // Create new vector of resampled particles
  std::vector<ParticleType<StateType>> tmp_particles(n_particles);
  for (ParticleType<StateType>& tmp_particle : tmp_particles) {
    double particle_random_num = uniform_dist(random_engine);

    for (int i = 0; i < n_particles; ++i) {
      if (particle_random_num < cum_sum[i]) {
        tmp_particle.first = particles->at(i).first;
      }
    }
  }

  // TODO(jwidauer): set particle weights
  *particles = tmp_particles;
}

}  // namespace refill

#endif  // REFILL_UTILITY_RESAMPLE_METHODS_H_
