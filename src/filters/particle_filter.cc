#include "refill/filters/particle_filter.h"

namespace refill {

void ParticleFilter::resample() {
  std::size_t n_particles = particles_.size();

  CHECK_NE(n_particles, 0)
      << "[ParticleFilter] Particle filter has not been initialized.";

  // Compute the cumulative sum of weights
  std::vector<double> cum_sum(n_particles);
  cum_sum[0] = particles_[0].weight;

  for (int i = 1; i < n_particles; ++i) {
    cum_sum[i] = cum_sum[i - 1] + particles_[i].weight;
  }

  std::random_device true_rng;
  std::default_random_engine random_engine(true_rng);
  std::uniform_real_distribution uniform_dist(0.0, 1.0);

  std::vector<double> random_vec(n_particles);
  std::generate(random_vec.begin(), random_vec.end(),
                uniform_dist(random_engine));
}

}  // namespace refill
