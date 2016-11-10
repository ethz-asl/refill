
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int state_dim = 2;
  constexpr int meas_dim  = 2;

  GaussianDistribution<state_dim> init_state;
  GaussianDistribution<state_dim> sys_noise;
  GaussianDistribution<meas_dim>  meas_noise;
  Eigen::MatrixXd sys_mat(state_dim, state_dim);
  Eigen::MatrixXd meas_mat(meas_dim, state_dim);

  sys_mat  = Eigen::MatrixXd::Identity(state_dim, state_dim);
  meas_mat = Eigen::MatrixXd::Identity(meas_dim, state_dim);

  LinearSystemModel<state_dim, 0> sys_mod(sys_mat,
                                       sys_noise);

  LinearMeasurementModel<state_dim, meas_dim> meas_mod(meas_mat,
                                                       meas_noise);

  KalmanFilter<state_dim, meas_dim> kf(init_state);

  kf.Predict(sys_mod);
  kf.Predict(sys_mod);

  ASSERT_EQ(kf.state().cov(),
            3.0 * Eigen::MatrixXd::Identity(state_dim, state_dim));

  Eigen::VectorXd measurement(meas_dim);

  measurement = Eigen::VectorXd::Zero(meas_dim);

  kf.Update(meas_mod, measurement);

  ASSERT_EQ(Eigen::VectorXd::Zero(state_dim), kf.state().mean());
}

}  // namespace refill
