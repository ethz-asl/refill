#ifndef INCLUDE_REFILL_FILTERS_KALMAN_FILTER_INL_H_
#define INCLUDE_REFILL_FILTERS_KALMAN_FILTER_INL_H_

#include <glog/logging.h>

namespace refill {

template <int SYSDIM, int MEASDIM>
KalmanFilter<SYSDIM, MEASDIM>::KalmanFilter() {
  constexpr int cur_sysdim = (SYSDIM == Eigen::Dynamic) ? 1 : SYSDIM;

  sys_model_ = Eigen::Matrix<double, cur_sysdim, cur_sysdim>::Identity(
      cur_sysdim, cur_sysdim);

  // If dimensionality of measurement datastructures has dynamic size, we assume
  // it to be the same as the system model by default.
  if (MEASDIM == Eigen::Dynamic) {
    measurement_model_ =
        Eigen::Matrix<double, cur_sysdim, cur_sysdim>::Identity(cur_sysdim,
                                                                cur_sysdim);
  } else {
    measurement_model_ = Eigen::Matrix<double, MEASDIM, cur_sysdim>::Identity(
        MEASDIM, cur_sysdim);
  }
}

template <int SYSDIM, int MEASDIM>
KalmanFilter<SYSDIM, MEASDIM>::KalmanFilter(
    GaussianDistribution<SYSDIM> initial_state,
    GaussianDistribution<SYSDIM> system_noise,
    GaussianDistribution<MEASDIM> measurement_noise,
    Eigen::Matrix<double, SYSDIM, SYSDIM> sys_model,
    Eigen::Matrix<double, MEASDIM, SYSDIM> obs_model)
    : state_(initial_state),
      system_noise_(system_noise),
      measurement_noise_(measurement_noise),
      sys_model_(sys_model),
      measurement_model_(obs_model) {
  const int state_dim = state_.dim();
  const int measurement_dim = measurement_noise.dim();

  if (SYSDIM == Eigen::Dynamic) {
    CHECK_EQ(state_dim, system_noise.dim());
    CHECK_EQ(state_dim, sys_model.cols());
    CHECK_EQ(state_dim, sys_model.rows());
    CHECK_EQ(state_dim, obs_model.cols());
  }

  if (MEASDIM == Eigen::Dynamic) {
    CHECK_EQ(measurement_dim, obs_model.rows());
    CHECK_EQ(measurement_dim, measurement_noise.dim());
  }
}

template <int SYSDIM, int MEASDIM>
void KalmanFilter<SYSDIM, MEASDIM>::Update(
    Eigen::Matrix<double, MEASDIM, 1> measurement) {
  CHECK_EQ(measurement.size(), measurement_model_.rows());

  Eigen::Matrix<double, MEASDIM, 1> innovation;
  Eigen::Matrix<double, MEASDIM, MEASDIM> residual_cov;
  Eigen::Matrix<double, SYSDIM, MEASDIM> kalman_gain;

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

#endif  // INCLUDE_REFILL_FILTERS_KALMAN_FILTER_INL_H_
