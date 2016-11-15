#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel() {
  constexpr int kCurrentSystemDIm =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;
  constexpr int kCurrentInputDIm =
      (INPUT_DIM == Eigen::Dynamic) ? 1 : INPUT_DIM;

  system_mat_ =
      Eigen::Matrix<double, kCurrentSystemDIm, kCurrentSystemDIm>::Identity(
          kCurrentSystemDIm, kCurrentSystemDIm);
  input_mat_ =
      Eigen::Matrix<double, kCurrentSystemDIm, kCurrentInputDIm>::Identity(
          kCurrentSystemDIm, kCurrentInputDIm);

  // In case of no declaration of system noise, we use standard normal gaussian.
  system_noise_.reset(new GaussianDistribution<kCurrentSystemDIm>());
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise,
    const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.Clone()) {
  const int state_dim = system_mat.rows();

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state_dim, system_mat.cols());
    CHECK_EQ(state_dim, input_mat.rows());
    CHECK_EQ(state_dim, system_noise.mean().size());
  }
}

template<int STATE_DIM, int INPUT_DIM>
Eigen::Matrix<double, STATE_DIM, 1>
  LinearSystemModel<STATE_DIM, INPUT_DIM>::Propagate(
    const Eigen::Matrix<double, STATE_DIM, 1>& state,
    const Eigen::Matrix<double, INPUT_DIM, 1>& input) const {

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state.size(), system_mat_.rows());
  }

  if (INPUT_DIM == Eigen::Dynamic) {
    CHECK_EQ(input.rows(), input_mat_.cols());
  }

  // If there is no input, we don't need to compute the matrix multiplication.
  if (input_mat_.cols() == 0) {
    return system_mat_ * state + system_noise_->mean();
  } else {
    return system_mat_ * state + input_mat_ * input + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
