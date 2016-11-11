#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int state_dim = 2;
  constexpr int meas_dim = 2;
  constexpr int input_dim = 2;

  GaussianDistribution<state_dim> init_state;
  GaussianDistribution<state_dim> sys_noise;
  GaussianDistribution<meas_dim> meas_noise;
  Eigen::MatrixXd sys_mat(state_dim, state_dim);
  Eigen::MatrixXd input_mat(state_dim, input_dim);
  Eigen::MatrixXd meas_mat(meas_dim, state_dim);

  sys_mat = Eigen::MatrixXd::Identity(state_dim, state_dim);
  input_mat = Eigen::MatrixXd::Identity(state_dim, input_dim);
  meas_mat = Eigen::MatrixXd::Identity(meas_dim, state_dim);

  LinearSystemModel<state_dim, input_dim> sys_mod(sys_mat, sys_noise,
                                                  input_mat);
  LinearMeasurementModel<state_dim, meas_dim> meas_mod(meas_mat, meas_noise);

  KalmanFilter<state_dim, meas_dim, input_dim> kf(init_state);

  Eigen::VectorXd input(input_dim);
  input = Eigen::VectorXd::Ones(input_dim);

  kf.Predict(sys_mod, input);
  kf.Predict(sys_mod, input);

  ASSERT_EQ(kf.state().cov(),
            3.0 * Eigen::MatrixXd::Identity(state_dim, state_dim));

  ASSERT_EQ(kf.state().mean(), 2.0 * Eigen::VectorXd::Ones(state_dim));

  Eigen::VectorXd measurement(meas_dim);

  measurement = 2.0 * Eigen::VectorXd::Ones(meas_dim);

  kf.Update(meas_mod, measurement);

  ASSERT_EQ(2.0 * Eigen::VectorXd::Ones(state_dim), kf.state().mean());
}

}  // namespace refill
