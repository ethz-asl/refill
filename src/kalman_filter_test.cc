#include "refill/filters/kalman_filter.h"

#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int state_dim = 2;
  constexpr int meas_dim  = 2;

  GaussianDistribution<state_dim> init_state;
  GaussianDistribution<state_dim> sys_noise;
  GaussianDistribution<meas_dim>  meas_noise;
  Eigen::MatrixXd sys_mod(state_dim, state_dim);
  Eigen::MatrixXd meas_mod(meas_dim, state_dim);

  sys_mod  = Eigen::MatrixXd::Identity(state_dim, state_dim);
  meas_mod = Eigen::MatrixXd::Identity(meas_dim, state_dim);

  KalmanFilter<state_dim, meas_dim> kf(init_state,
                                       sys_noise,
                                       meas_noise,
                                       sys_mod,
                                       meas_mod);
  kf.Predict();
  kf.Predict();

  ASSERT_EQ(kf.state().cov(),
            3.0 * Eigen::MatrixXd::Identity(state_dim, state_dim));

  Eigen::VectorXd measurement(meas_dim);

  measurement = Eigen::VectorXd::Zero(meas_dim);

  kf.Update(measurement);

  ASSERT_EQ(Eigen::VectorXd::Zero(state_dim), kf.state().mean());
}

}  // namespace refill
