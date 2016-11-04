#include "refill/filters/kalman_filter.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int s_dim = 2;
  constexpr int m_dim  = 2;

  GaussianDistribution<s_dim> init_state;
  GaussianDistribution<s_dim> sys_noise;
  GaussianDistribution<m_dim>  meas_noise;
  Eigen::Matrix<double, s_dim, s_dim> sys_mod;
  Eigen::Matrix<double, m_dim, s_dim> meas_mod;

  sys_mod  = Eigen::Matrix<double, s_dim, s_dim>::Identity(s_dim, s_dim);
  meas_mod = Eigen::Matrix<double, m_dim, s_dim>::Identity(m_dim, s_dim);

  KalmanFilter<s_dim, m_dim> kf(init_state,
                                sys_noise,
                                meas_noise,
                                sys_mod,
                                meas_mod);
  kf.Predict();
  kf.Predict();

  ASSERT_EQ(kf.state().cov(),
            3.0 * Eigen::MatrixXd::Identity(s_dim, s_dim));

  kf.Update(Eigen::VectorXd::Zero(m_dim));

  ASSERT_EQ(Eigen::VectorXd::Zero(s_dim), kf.state().mean());
}

}  // namespace refill
