#ifndef REFILL_FILTERS_KALMAN_FILTER_INL_H_
#define REFILL_FILTERS_KALMAN_FILTER_INL_H_

#include <glog/logging.h>

namespace refill {

template <int STATE_DIM, int MEAS_DIM>
KalmanFilter<STATE_DIM, MEAS_DIM>::KalmanFilter() {
  constexpr int cur_sysdim = (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;

  sys_model_ = Eigen::Matrix<double, cur_sysdim, cur_sysdim>::Identity(
      cur_sysdim, cur_sysdim);

  // If dimensionality of measurement datastructures has dynamic size, we assume
  // it to be the same as the system model by default.
  if (MEAS_DIM == Eigen::Dynamic) {
    measurement_model_ =
        Eigen::Matrix<double, cur_sysdim, cur_sysdim>::Identity(cur_sysdim,
                                                                cur_sysdim);
  } else {
    measurement_model_ = Eigen::Matrix<double, MEAS_DIM, cur_sysdim>::Identity(
        MEAS_DIM, cur_sysdim);
  }
}

template <int STATE_DIM, int MEAS_DIM>
KalmanFilter<STATE_DIM, MEAS_DIM>::KalmanFilter(
    GaussianDistribution<STATE_DIM> initial_state,
    GaussianDistribution<STATE_DIM> system_noise,
    GaussianDistribution<MEAS_DIM> measurement_noise,
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> sys_model,
    Eigen::Matrix<double, MEAS_DIM, STATE_DIM> obs_model)
    : state_(initial_state),
      system_noise_(system_noise),
      measurement_noise_(measurement_noise),
      sys_model_(sys_model),
      measurement_model_(obs_model) {
  const int state_dim = state_.dim();
  const int measurement_dim = measurement_noise.dim();

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state_dim, system_noise.dim());
    CHECK_EQ(state_dim, sys_model.cols());
    CHECK_EQ(state_dim, sys_model.rows());
    CHECK_EQ(state_dim, obs_model.cols());
  }

  if (MEAS_DIM == Eigen::Dynamic) {
    CHECK_EQ(measurement_dim, obs_model.rows());
    CHECK_EQ(measurement_dim, measurement_noise.dim());
  }
}

template <int STATE_DIM, int MEAS_DIM>
void KalmanFilter<STATE_DIM, MEAS_DIM>::Update(
    Eigen::Matrix<double, MEAS_DIM, 1> measurement) {
  CHECK_EQ(measurement.size(), measurement_model_.rows());

  Eigen::Matrix<double, MEAS_DIM, 1> innovation;
  Eigen::Matrix<double, MEAS_DIM, MEAS_DIM> residual_cov;
  Eigen::Matrix<double, STATE_DIM, MEAS_DIM> kalman_gain;

  innovation = measurement - measurement_model_ * state_.mean();
  residual_cov =
      measurement_model_ * state_.cov() * measurement_model_.transpose() +
      measurement_noise_.cov();
  kalman_gain =
      state_.cov() * measurement_model_.transpose() * residual_cov.inverse();

  state_.SetDistParam(
      state_.mean() + kalman_gain * innovation,
      state_.cov() - kalman_gain * residual_cov * kalman_gain.transpose());
}

}  // namespace refill

#endif  // REFILL_FILTERS_KALMAN_FILTER_INL_H_
