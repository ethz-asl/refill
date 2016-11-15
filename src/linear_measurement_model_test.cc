#include "refill/measurement_models/linear_measurement_model.h"

#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace refill {

TEST(LinearMeasurementModelTest, Fullrun) {
  constexpr int kStateDim = 2;
  constexpr int kMeasurementDim = 1;

  // Set up measurement matrix
  Eigen::MatrixXd measurement_mat(kMeasurementDim, kStateDim);
  measurement_mat = Eigen::MatrixXd::Identity(kMeasurementDim, kStateDim);

  // Set up measurement noise
  GaussianDistribution<kMeasurementDim> measurement_noise;
  measurement_noise.SetDistParam(
      Eigen::VectorXd::Ones(kMeasurementDim),
      Eigen::MatrixXd::Identity(kMeasurementDim, kMeasurementDim));

  // Set up linear measurement model
  LinearMeasurementModel<kStateDim, kMeasurementDim> measurement_model(
      measurement_mat, measurement_noise);

  // Set up state vector for observation
  Eigen::VectorXd state(kStateDim);
  state = Eigen::VectorXd::Ones(kStateDim);

  // Check if observation works
  ASSERT_EQ(measurement_model.Observe(state),
            Eigen::VectorXd::Ones(kMeasurementDim) * 2.0);

  // Check if dimension getters work
  ASSERT_EQ(measurement_model.GetStateDim(), kStateDim);
  ASSERT_EQ(measurement_model.GetMeasurementDim(), kMeasurementDim);

  // Check if Jacobian getter works
  ASSERT_EQ(measurement_model.GetJacobian(), measurement_mat);
}

}  // namespace refill
