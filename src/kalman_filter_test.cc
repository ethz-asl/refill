#include "refill/kalman_filter.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  KalmanFilter kf;
  kf.Predict();
  kf.Predict();

  ASSERT_EQ(kf.state().cov(), 3.0 * Eigen::MatrixXd::Identity(1, 1));

  kf.Update(Eigen::VectorXd::Zero(1));

  ASSERT_EQ(Eigen::VectorXd::Zero(1), kf.state().mean());
}

}  // namespace refill
