#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/extended_kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(ExtendedKalmanFilterTest, FullRun) {
  constexpr int kStateDim = 3;
  constexpr int kMeasurementDim = 3;
  constexpr int kInputDim = 3;

  // Set up initial state, as well as system and measurement noise
  GaussianDistribution initial_state(kStateDim);
  GaussianDistribution system_noise(kStateDim);
  GaussianDistribution measurement_noiseXd(kMeasurementDim);

  // Set up system, input and measurement matrix
  const Eigen::MatrixXd system_mat = Eigen::MatrixXd::Identity(kStateDim,
                                                               kStateDim);
  const Eigen::MatrixXd input_mat = Eigen::MatrixXd::Identity(kStateDim,
                                                              kInputDim);
  const Eigen::MatrixXd measurement_mat = Eigen::MatrixXd::Identity(
      kMeasurementDim, kStateDim);

  // Set up linear system and measurement model
  LinearSystemModel system_model(system_mat, system_noise, input_mat);
  LinearMeasurementModel measurement_model(measurement_mat,
                                           measurement_noiseXd);

  // Set up kalman filter
  ExtendedKalmanFilter KF(initial_state);

  // Set up input and measurement vectors
  const Eigen::VectorXd input = Eigen::VectorXd::Ones(kInputDim);
  const Eigen::VectorXd measurement = Eigen::VectorXd::Ones(kMeasurementDim);

  // Perform one prediction step
  KF.predict(system_model, input);

  // Check if prediction was correct
  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));
  ASSERT_EQ(KF.state().cov(),
            Eigen::MatrixXd::Identity(kStateDim, kStateDim) * 2.0);

  // Perform one update step
  KF.update(measurement_model, measurement);

  // Check if update was correct
  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
