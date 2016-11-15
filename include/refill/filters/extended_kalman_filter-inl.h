#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_INL_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATE_DIM>
ExtendedKalmanFilter<STATE_DIM>::ExtendedKalmanFilter() {
}

template<int STATE_DIM>
ExtendedKalmanFilter<STATE_DIM>::ExtendedKalmanFilter(
    const GaussianDistribution<STATE_DIM>& initial_state)
    : state_(initial_state) {
}

template<int STATE_DIM>
template<int INPUT_DIM>
void ExtendedKalmanFilter<STATE_DIM>::Predict(
    const SystemModelBase<STATE_DIM, INPUT_DIM>& system_model,
    const Eigen::Matrix<double, INPUT_DIM, 1>& input) {

  Eigen::Matrix<double, STATE_DIM, STATE_DIM> system_mat;
  system_mat = system_model.GetJacobian();

  state_.SetDistParam(
      system_model.Propagate(state_.mean(), input),
      system_mat * state_.cov() * system_mat.transpose()
          + system_model.GetSystemNoise()->cov());
}

template<int STATE_DIM>
template<int MEAS_DIM>
void ExtendedKalmanFilter<STATE_DIM>::Update(
    const MeasurementModelBase<STATE_DIM, MEAS_DIM>& measurement_model,
    const Eigen::Matrix<double, MEAS_DIM, 1>& measurement) {
  CHECK_EQ(measurement.size(), measurement_model.GetMeasurementDim());

  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> measurement_mat;
  measurement_mat = measurement_model.GetJacobian();

  if (MEAS_DIM == Eigen::Dynamic) {
    CHECK_EQ(measurement.size(), measurement_model.GetMeasurementDim());
  }

  Eigen::Matrix<double, MEAS_DIM, 1> innovation;
  Eigen::Matrix<double, MEAS_DIM, MEAS_DIM> residual_cov;
  Eigen::Matrix<double, STATE_DIM, MEAS_DIM> kalman_gain;

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
