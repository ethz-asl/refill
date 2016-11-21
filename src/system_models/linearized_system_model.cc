#include "refill/system_models/linearized_system_model.h"

namespace refill {

LinearizedSystemModel::LinearizedSystemModel(
    const unsigned int& state_dim, const DistributionInterface& system_noise)
    : LinearizedSystemModel(state_dim, system_noise, 0) {}

LinearizedSystemModel::LinearizedSystemModel(
    const unsigned int& state_dim, const DistributionInterface& system_noise,
    const unsigned int& input_dim)
    : state_dim_(state_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {}

void LinearizedSystemModel::setLinearizedSystemParameters(
    const unsigned int& state_dim, const DistributionInterface& system_noise) {
  this->setLinearizedSystemParameters(state_dim, system_noise, 0);
}

void LinearizedSystemModel::setLinearizedSystemParameters(
    const unsigned int& state_dim, const DistributionInterface& system_noise,
    const unsigned int& input_dim) {
  state_dim_ = state_dim;
  system_noise_.reset(system_noise.clone());
  input_dim_ = input_dim;
}

unsigned int LinearizedSystemModel::getStateDim() const {
  return state_dim_;
}

unsigned int LinearizedSystemModel::getSystemNoiseDim() const {
  CHECK(system_noise_)
      << "[LinearizedSystemModel] System noise has not been set.";
  return system_noise_->mean().size();
}

unsigned int LinearizedSystemModel::getInputDim() const {
  return input_dim_;
}

DistributionInterface* LinearizedSystemModel::getSystemNoise() const {
  CHECK(system_noise_)
      << "[LinearizedSystemModel] System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
