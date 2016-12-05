#include "refill/filters/particle_filter.h"

namespace refill {

void ParticleFilter::resample() {
  resample_method_(particles_);
}

}  // namespace refill
