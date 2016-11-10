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
    const DistributionBase<STATEDIM>& system_noise,
    const Eigen::Matrix<double, STATEDIM, INPUTDIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.Clone()) {
}

template<int STATEDIM, int INPUTDIM>
void LinearSystemModel<STATEDIM, INPUTDIM>::Propagate(
    Eigen::Matrix<double, STATEDIM, 1>* state,
    const Eigen::Matrix<double, INPUTDIM, 1> &input) {

  // If there is no input, we don't need to compute the matrix multiplication.
  if (INPUTDIM == 0) {
    *state = system_mat_ * (*state) + system_noise_->mean();
  } else {
    *state = system_mat_ * (*state) + input_mat_ * input
        + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
