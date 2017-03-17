#include "refill/system_models/system_model_base.h"

using std::size_t;

namespace refill {


/**
 * Use this constructor if your system does not have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 */
SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0u) {}


/**
 * Use this constructor if your system model does have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 * @param input_dim The systems input dimension.
 */
SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise,
                                 const size_t& input_dim)
    : state_dim_(state_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {}

Eigen::MatrixXd SystemModelBase::propagateVectorized(
    const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
    const Eigen::MatrixXd& sampled_noise) const {
  CHECK_EQ(getStateDim(), sampled_state.rows());
  CHECK_EQ(getSystemNoiseDim(), sampled_noise.rows());

  if (getInputDim() != 0) {
    CHECK_EQ(getInputDim(), input.rows());
  }

  const size_t state_size = getStateDim();
  const size_t noise_size = getSystemNoiseDim();
  const size_t state_sample_count = sampled_state.cols();
  const size_t noise_sample_count = sampled_noise.cols();

  Eigen::MatrixXd result(state_size, state_sample_count * noise_sample_count);

  // Evaluate the propagate function for each combination of state / noise
  // samples.
  for (size_t i = 0u; i < state_sample_count; ++i) {
    for (size_t j = 0u; j < noise_sample_count; ++j) {
      result.col(i * noise_sample_count + j) = propagate(sampled_state.col(i),
                                                         input,
                                                         sampled_noise.col(j));
    }
  }

  return result;
}

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

/** @return the state dimension. */
size_t SystemModelBase::getStateDim() const {
  return state_dim_;
}

/** @return the input dimension. */
size_t SystemModelBase::getInputDim() const {
  return input_dim_;
}

/** @return the noise dimension. */
size_t SystemModelBase::getSystemNoiseDim() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_->mean().size();
}

/** @return a pointer to the system noise distribution. */
DistributionInterface* SystemModelBase::getSystemNoise() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
