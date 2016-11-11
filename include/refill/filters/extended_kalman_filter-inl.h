#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_INL_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATEDIM>
ExtendedKalmanFilter<STATEDIM>::ExtendedKalmanFilter() {
}

template<int STATEDIM>
ExtendedKalmanFilter<STATEDIM>::ExtendedKalmanFilter(
    const GaussianDistribution<STATEDIM>& initial_state)
    : state_(initial_state) {
}

template<int STATEDIM>
template<int INPUTDIM>
void ExtendedKalmanFilter<STATEDIM>::Predict(
    const SystemModelBase<STATEDIM, INPUTDIM>& system_model,
    const Eigen::Matrix<double, INPUTDIM, 1>& input) {

  Eigen::Matrix<double, STATEDIM, STATEDIM> system_mat;
  system_mat = system_model.GetJacobian();

  state_.SetDistParam(
      system_model.Propagate(state_.mean(), input),
      system_mat * state_.cov() * system_mat.transpose()
          + system_model.GetSystemNoise()->cov());
}

template<int STATEDIM>
template<int MEASDIM>
void ExtendedKalmanFilter<STATEDIM>::Update(
    const MeasurementModelBase<STATEDIM, MEASDIM>& measurement_model,
    const Eigen::Matrix<double, MEASDIM, 1>& measurement) {
  CHECK_EQ(measurement.size(), measurement_model.GetMeasurementDim());

  Eigen::Matrix<double, MEASDIM, STATEDIM> measurement_mat;
  measurement_mat = measurement_model.GetJacobian();

  if (MEASDIM == Eigen::Dynamic) {
    CHECK_EQ(measurement.size(), measurement_model.GetMeasurementDim());
  }

  Eigen::Matrix<double, MEASDIM, 1> innovation;
  Eigen::Matrix<double, MEASDIM, MEASDIM> residual_cov;
  Eigen::Matrix<double, STATEDIM, MEASDIM> kalman_gain;

  innovation = measurement - measurement_model.Observe(state_.mean());
  residual_cov = measurement_mat * state_.cov() * measurement_mat.transpose()
      + measurement_model.GetMeasurementNoise()->cov();
  kalman_gain = state_.cov() * measurement_mat.transpose()
      * residual_cov.inverse();

  state_.SetDistParam(
      state_.mean() + kalman_gain * innovation,
      state_.cov() - kalman_gain * residual_cov * kalman_gain.transpose());
}

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_INL_H_
