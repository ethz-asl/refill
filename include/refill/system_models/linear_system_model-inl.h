#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATEDIM, int INPUTDIM>
LinearSystemModel<STATEDIM, INPUTDIM>::LinearSystemModel() {
  constexpr int cur_sysdim = (STATEDIM == Eigen::Dynamic) ? 1 : STATEDIM;
  constexpr int cur_indim = (INPUTDIM == Eigen::Dynamic) ? 1 : INPUTDIM;

  system_mat_ = Eigen::Matrix<double, cur_sysdim, cur_sysdim>::Identity(
      cur_sysdim, cur_sysdim);
  input_mat_ = Eigen::Matrix<double, cur_sysdim, cur_indim>::Identity(
      cur_sysdim, cur_indim);

  // In case of no declaration of system noise, we use standard normal gaussian.
  system_noise_.reset(new GaussianDistribution<cur_sysdim>());
}

template<int STATEDIM, int INPUTDIM>
LinearSystemModel<STATEDIM, INPUTDIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATEDIM, STATEDIM>& system_mat,
    const DistributionInterface<STATEDIM>& system_noise,
    const Eigen::Matrix<double, STATEDIM, INPUTDIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.Clone()) {
  const int state_dim = system_mat.rows();

  if (STATEDIM == Eigen::Dynamic) {
    CHECK_EQ(state_dim, system_mat.cols());
    CHECK_EQ(state_dim, input_mat.rows());
    CHECK_EQ(state_dim, system_noise.mean().rows());
  }
}

template<int STATEDIM, int INPUTDIM>
Eigen::Matrix<double, STATEDIM, 1>
  LinearSystemModel<STATEDIM, INPUTDIM>::Propagate(
    const Eigen::Matrix<double, STATEDIM, 1>& state,
    const Eigen::Matrix<double, INPUTDIM, 1>& input) const {

  // If there is no input, we don't need to compute the matrix multiplication.
  if (INPUTDIM == 0) {
    return system_mat_ * state + system_noise_->mean();
  } else {
    return system_mat_ * state + input_mat_ * input + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
