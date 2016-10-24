#include "refill/kalman_filter.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>

namespace refill {

KalmanFilter::KalmanFilter() {
  sys_model_ = Eigen::MatrixXd::Identity(1, 1);
  measurement_model_ = Eigen::MatrixXd::Identity(1, 1);
}

KalmanFilter::KalmanFilter(GaussianDistribution state,
                           GaussianDistribution system_noise,
                           GaussianDistribution measurement_noise,
                           Eigen::MatrixXd sys_model, Eigen::MatrixXd obs_model)
    : state_(state),
      system_noise_(system_noise),
      measurement_noise_(measurement_noise),
      sys_model_(sys_model),
      measurement_model_(obs_model) {
  const int state_dim = state.dim();
  const int measurement_dim = measurement_noise.dim();

  CHECK_EQ(state_dim, system_noise.dim());
  CHECK_EQ(state_dim, sys_model.cols());
  CHECK_EQ(state_dim, sys_model.rows());
  CHECK_EQ(state_dim, obs_model.cols());
  CHECK_EQ(measurement_dim, obs_model.rows());
  CHECK_EQ(measurement_dim, measurement_noise.dim());
}

void KalmanFilter::Update(Eigen::VectorXd measurement) {
  CHECK_EQ(measurement.size(), measurement_model_.rows());

  Eigen::VectorXd innovation;
  Eigen::MatrixXd residual_cov;
  Eigen::MatrixXd kalman_gain;

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
