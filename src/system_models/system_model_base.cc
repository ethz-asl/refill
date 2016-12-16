#include "refill/system_models/system_model_base.h"

using std::size_t;

namespace refill {

SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0u) {}

SystemModelBase::SystemModelBase(const size_t& state_dim,
                                 const DistributionInterface& system_noise,
                                 const size_t& input_dim)
    : state_dim_(state_dim),
      input_dim_(input_dim),
      system_noise_(system_noise.clone()) {}

Eigen::MatrixXd SystemModelBase::propagateVectorized(
    const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
    const Eigen::MatrixXd& sampled_noise) {
  const size_t state_size = getStateDim();
  const size_t noise_size = getSystemNoiseDim();
  const size_t state_sample_count = sampled_state.cols();
  const size_t noise_sample_count = sampled_noise.cols();

  CHECK_EQ(sampled_state.rows(), state_size);
  CHECK_EQ(input.size(), getInputDim());
  CHECK_EQ(sampled_noise.rows(), getSystemNoiseDim());

  Eigen::MatrixXd result(state_size, state_sample_count * noise_sample_count);

  // Evaluate the propagate function for each combination of state / noise
  // samples.
  for (size_t state_sample_id = 0u; state_sample_id < state_sample_count;
       state_sample_id++) {
    for (size_t noise_sample_id = 0u; noise_sample_id < noise_sample_count;
         noise_sample_id++) {
      result.block(0, state_sample_id * noise_sample_count + noise_sample_id,
                   state_size, 1) =
          propagate(sampled_state.block(0, state_sample_id, 1, state_size),
                    input,
                    sampled_noise.block(0, noise_sample_id, 1, noise_size));
    }
  }

  return result;
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

size_t SystemModelBase::getStateDim() const { return state_dim_; }

size_t SystemModelBase::getInputDim() const { return input_dim_; }

size_t SystemModelBase::getSystemNoiseDim() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_->mean().size();
}

DistributionInterface* SystemModelBase::getSystemNoise() const {
  CHECK(system_noise_) << "System noise has not been set.";
  return system_noise_.get();
}

}  // namespace refill
