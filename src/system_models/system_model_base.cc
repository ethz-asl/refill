#include "refill/system_models/system_model_base.h"

namespace refill {

SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {
}

SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise,
                                 const size_t& input_dim)
    : state_dim_(state_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {
}

void SystemModelBase::setSystemModelBaseParameters(
    const size_t& state_dim, const DistributionInterface& system_noise) {
  this->setSystemModelBaseParameters(state_dim, system_noise, 0);
}

void SystemModelBase::setSystemModelBaseParameters(
    const size_t& state_dim, const DistributionInterface& system_noise,
    const size_t& input_dim) {
  state_dim_ = state_dim;
  system_noise_.reset(system_noise.clone());
  input_dim_ = input_dim;
}

size_t SystemModelBase::getStateDim() const {
  return state_dim_;
}

size_t SystemModelBase::getInputDim() const {
  return input_dim_;
}

size_t SystemModelBase::getSystemNoiseDim() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_->mean().size();
}

DistributionInterface* SystemModelBase::getSystemNoise() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
