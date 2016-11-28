#include "refill/system_models/system_model_base.h"

#include <cstddef>
#include "Eigen/Dense"

namespace refill {

Eigen::MatrixXd SystemModelBase::propagateVectorized(
    const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
    const Eigen::VectorXd& sampled_noise) {
  const size_t state_size = getStateDim();
  const size_t state_sample_count = sampled_state.cols();
  const size_t noise_sample_count = sampled_noise.cols();

  CHECK_EQ(sampled_state.rows(), state_size);
  CHECK_EQ(input.size(), getInputDim());
  CHECK_EQ(sampled_noise.rows(), getSystemNoiseDim());

  Eigen::MatrixXd result(state_size, state_sample_count * noise_sample_count);

  // Evaluate the propagate function for each combination of state / noise
  // samples.
  for (int state_sample_id = 0; state_sample_id < state_sample_count;
       state_sample_id++) {
    for (int noise_sample_id = 0; noise_sample_id < noise_sample_count;
         noise_sample_id++) {
      // TODO(igilitschenski): FILLME
    }
  }

  return result;
}

}  // namespace refill
