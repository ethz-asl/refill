#include "refill/system_models/linearized_system_model.h"

namespace refill {

LinearizedSystemModel::LinearizedSystemModel() {
}

LinearizedSystemModel::LinearizedSystemModel(
    const int& system_dim, const DistributionInterface& system_noise)
    : LinearizedSystemModel(system_dim, system_noise, 0) {
}

LinearizedSystemModel::LinearizedSystemModel(
    const int& system_dim, const DistributionInterface& system_noise,
    const int& input_dim)
    : system_dim_(system_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {
}

int LinearizedSystemModel::getSystemNoiseDim() const {
  CHECK(system_noise_)
      << "[LinearizedSystemModel] System noise has not been set.";
  return system_noise_->mean().size();
}

DistributionInterface* LinearizedSystemModel::getSystemNoise() const {
  CHECK(system_noise_)
      << "[LinearizedSystemModel] System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
