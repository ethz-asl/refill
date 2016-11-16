#include "refill/filters/kalman_filter.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int kStateDim = 2;
  constexpr int kMeasurementDim  = 2;

  GaussianDistribution<kStateDim> init_state;
  GaussianDistribution<kStateDim> sys_noise;
  GaussianDistribution<kMeasurementDim>  meas_noise;
  Eigen::MatrixXd sys_mod(kStateDim, kStateDim);
  Eigen::MatrixXd meas_mod(kMeasurementDim, kStateDim);

  sys_mod  = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
  meas_mod = Eigen::MatrixXd::Identity(kMeasurementDim, kStateDim);

  KalmanFilter<kStateDim, kMeasurementDim> kf(init_state,
                                       sys_noise,
                                       meas_noise,
                                       sys_mod,
                                       meas_mod);
  kf.predict();
  kf.predict();

  ASSERT_EQ(kf.state().cov(),
            3.0 * Eigen::MatrixXd::Identity(kStateDim, kStateDim));

  Eigen::VectorXd measurement(kMeasurementDim);

  measurement = Eigen::VectorXd::Zero(kMeasurementDim);

  kf.update(measurement);

  ASSERT_EQ(Eigen::VectorXd::Zero(kStateDim), kf.state().mean());
}

}  // namespace refill
