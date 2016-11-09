#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_

#include <glog/logging.h>

namespace refill {

template <int STATEDIM, int INPUTDIM>
LinearSystemModel<STATEDIM, INPUTDIM>::LinearSystemModel() {
  // In case of dynamic state and input size, we use a univariate system
  // with no input and univariate standard normal gaussian distribution.
  if (STATEDIM == Eigen::Dynamic && INPUTDIM == Eigen::Dynamic) {
    system_mat_ = Eigen::MatrixXd::Identity(1, 1);
    input_mat_  = Eigen::MatrixXd::Zero(1);
    system_noise_ = new GaussianDistribution<1>();
  }
}

template <int STATEDIM, int INPUTDIM>
LinearSystemModel<STATEDIM, INPUTDIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATEDIM, STATEDIM>& system_mat,
    const Eigen::Matrix<double, STATEDIM, INPUTDIM>& input_mat,
    const DistributionBase<STATEDIM>& system_noise)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.Clone()) {
}

template <int STATEDIM, int INPUTDIM>
void LinearSystemModel<STATEDIM, INPUTDIM>::Propagate(
    Eigen::Matrix<double, STATEDIM, 1>* state,
    const Eigen::Matrix<double, INPUTDIM, 1> &input) {
  *state = system_mat_ * (*state) + input_mat_ * input + system_noise_->mean();
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_INL_H_
