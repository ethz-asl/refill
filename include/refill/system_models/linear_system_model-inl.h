#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel() {
  constexpr int kCurrentSystemDim =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;
  constexpr int kCurrentInputDim =
      (INPUT_DIM == Eigen::Dynamic) ? 1 : INPUT_DIM;

  system_mat_ =
      Eigen::Matrix<double, kCurrentSystemDim, kCurrentSystemDim>::Identity(
          kCurrentSystemDim, kCurrentSystemDim);
  input_mat_ =
      Eigen::Matrix<double, kCurrentSystemDim, kCurrentInputDim>::Identity(
          kCurrentSystemDim, kCurrentInputDim);

  // In case of no declaration of system noise, we use standard normal gaussian
  system_noise_.reset(new GaussianDistribution<kCurrentSystemDim>());
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise,
    const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.Clone()) {
}

template<int STATE_DIM, int INPUT_DIM>
void LinearSystemModel<STATE_DIM, INPUT_DIM>::Propagate(
    Eigen::Matrix<double, STATE_DIM, 1>* state,
    const Eigen::Matrix<double, INPUT_DIM, 1> &input) {

  // If there is no input, we don't need to compute the matrix multiplication.
  if (INPUT_DIM == 0) {
    *state = system_mat_ * (*state) + system_noise_->mean();
  } else {
    *state = system_mat_ * (*state) + input_mat_ * input
        + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
