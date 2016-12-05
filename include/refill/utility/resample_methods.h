#ifndef REFILL_UTILITY_RESAMPLE_METHODS_H_
#define REFILL_UTILITY_RESAMPLE_METHODS_H_

#include <glog/logging.h>

#include <utility>
#include <random>
#include <vector>

template<typename ParticleType>
using ParticleVectorType = std::vector<std::pair<ParticleType, double>>;

namespace refill {

template<typename ParticleType>
void importance_sampling(ParticleVectorType<ParticleType>* particles) {
  std::size_t n_particles = particles->size();

  CHECK_NE(n_particles, 0)<< "Particle vector is empty.";

  // Compute the cumulative sum of weights
  std::vector<double> cum_sum(n_particles);
  cum_sum[0] = particles->at(0).second;

  for (int i = 1; i < n_particles; ++i) {
    cum_sum[i] = cum_sum[i - 1] + particles->at(i).second;
  }

  std::random_device true_rng;
  std::default_random_engine random_engine(true_rng);
  std::uniform_real_distribution uniform_dist(0.0, 1.0);

  std::vector<double> random_vec(n_particles);
  std::generate(random_vec.begin(), random_vec.end(),
                uniform_dist(random_engine));

  ParticleVectorType<ParticleType> tmp_particles(n_particles);
  for (int i = 0; i < n_particles; ++i) {
    for (int j = 0; j < n_particles; ++j) {
      if (random_vec[i] < cum_sum[j]) {
        tmp_particles[i].first = particles->at(j).first;
      }
    }
  }
}

}  // namespace refill

#endif  // REFILL_UTILITY_RESAMPLE_METHODS_H_
