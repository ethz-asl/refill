#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_INL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATEDIM, int MEASDIM>
LinearMeasurementModel<STATEDIM, MEASDIM>::LinearMeasurementModel() {
  constexpr int cur_sysdim = (STATEDIM == Eigen::Dynamic) ? 1 : STATEDIM;
  constexpr int cur_measdim = (MEASDIM == Eigen::Dynamic) ? 1 : MEASDIM;

  measurement_mat_ = Eigen::Matrix<double, cur_measdim, cur_sysdim>::Identity(
      cur_measdim, cur_sysdim);

  // In case of no declaration of measurement noise,
  // we use standard normal gaussian.
  measurement_noise_.reset(new GaussianDistribution<cur_measdim>());
}

template<int STATEDIM, int MEASDIM>
LinearMeasurementModel<STATEDIM, MEASDIM>::LinearMeasurementModel(
    const Eigen::Matrix<double, MEASDIM, STATEDIM>& measurement_mat,
    const DistributionInterface<MEASDIM>& measurement_noise)
    : measurement_mat_(measurement_mat),
      measurement_noise_(measurement_noise.Clone()) {

  if (MEASDIM == Eigen::Dynamic) {
    CHECK_EQ(measurement_mat_.rows(), measurement_noise_->mean().size());
  }
}

template<int STATEDIM, int MEASDIM>
Eigen::Matrix<double, MEASDIM, 1>
  LinearMeasurementModel<STATEDIM, MEASDIM>::Observe(
    const Eigen::Matrix<double, STATEDIM, 1>& state) const {
  if (STATEDIM == Eigen::Dynamic) {
    CHECK_EQ(state.size(), measurement_mat_.cols());
  }

  return measurement_mat_ * state + measurement_noise_->mean();
}

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_INL_H_
