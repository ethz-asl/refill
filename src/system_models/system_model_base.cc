#include "refill/system_models/system_model_base.h"

namespace refill {

/**
 * Use this constructor if your system does not have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 */
SystemModelBase::SystemModelBase(const std::size_t& state_dim,
                                 const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {}

/**
 * Use this constructor if your system model does have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 * @param input_dim The systems input dimension.
 */
SystemModelBase::SystemModelBase(const std::size_t& state_dim,
                                 const DistributionInterface& system_noise,
                                 const std::size_t& input_dim)
    : state_dim_(state_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {}

/**
 * Use this function if your system does not have an input.
 * The function clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 */
void SystemModelBase::setSystemModelBaseParameters(
    const size_t& state_dim, const DistributionInterface& system_noise) {
  this->setSystemModelBaseParameters(state_dim, system_noise, 0);
}

/**
 * Use this function if your system model does have an input.
 * The function clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 * @param input_dim The systems input dimension.
 */
void SystemModelBase::setSystemModelBaseParameters(
    const size_t& state_dim, const DistributionInterface& system_noise,
    const size_t& input_dim) {
  state_dim_ = state_dim;
  system_noise_.reset(system_noise.clone());
  input_dim_ = input_dim;
}

/**
 * @return the state dimension.
 */
std::size_t SystemModelBase::getStateDim() const {
  return state_dim_;
}

/**
 * @return the input dimension.
 */
std::size_t SystemModelBase::getInputDim() const {
  return input_dim_;
}

/**
 * @return the noise dimension.
 */
std::size_t SystemModelBase::getSystemNoiseDim() const {
  CHECK(system_noise_) << "[SystemModelBase] System noise has not been set.";
  return system_noise_->mean().size();
}

/**
 * @return a pointer to the system noise distribution.
 */
DistributionInterface* SystemModelBase::getSystemNoise() const {
  CHECK(system_noise_) << "[SystemModelBase] System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
