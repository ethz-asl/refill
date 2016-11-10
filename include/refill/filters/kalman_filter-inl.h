#ifndef REFILL_FILTERS_KALMAN_FILTER_INL_H_
#define REFILL_FILTERS_KALMAN_FILTER_INL_H_

#include <glog/logging.h>

namespace refill {

template<int STATEDIM, int MEASDIM, int INPUTDIM>
KalmanFilter<STATEDIM, MEASDIM, INPUTDIM>::KalmanFilter() {
}

template<int STATEDIM, int MEASDIM, int INPUTDIM>
KalmanFilter<STATEDIM, MEASDIM, INPUTDIM>::KalmanFilter(
    const GaussianDistribution<STATEDIM>& initial_state)
    : state_(initial_state) {
}

template<int STATEDIM, int MEASDIM, int INPUTDIM>
void KalmanFilter<STATEDIM, MEASDIM, INPUTDIM>::Predict(
    const LinearSystemModel<STATEDIM, INPUTDIM>& system_model,
    const Eigen::Matrix<double, INPUTDIM, 1>& input) {
  if (STATEDIM == Eigen::Dynamic) {
    CHECK_EQ(state_.dim(), system_model.dim());
  }

  Eigen::Matrix<double, STATEDIM, STATEDIM> system_mat;
  system_mat = system_model.GetSystemMatrix();

  state_.SetDistParam(
      system_model.Propagate(state_.mean(), input),
      system_mat * state_.cov() * system_mat.transpose()
          + system_model.GetSystemNoise()->cov());
}

template<int STATEDIM, int MEASDIM, int INPUTDIM>
void KalmanFilter<STATEDIM, MEASDIM, INPUTDIM>::Update(
    const LinearMeasurementModel<STATEDIM, MEASDIM>& measurement_model,
    const Eigen::Matrix<double, MEASDIM, 1>& measurement) {
  CHECK_EQ(measurement.size(), measurement_model.dim());

  Eigen::Matrix<double, MEASDIM, STATEDIM> measurement_mat;
  measurement_mat = measurement_model.GetMeasurementMatrix();

  if (STATEDIM == Eigen::Dynamic) {
    CHECK_EQ(state_.dim(), measurement_mat.cols());
  }

  if (MEASDIM == Eigen::Dynamic) {
    CHECK_EQ(measurement.rows(), measurement_mat.rows());
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

#endif  // REFILL_FILTERS_KALMAN_FILTER_INL_H_
