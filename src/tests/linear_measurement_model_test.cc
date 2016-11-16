#include "refill/measurement_models/linear_measurement_model.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(LinearMeasurementModelTest, Fullrun) {
  constexpr int kStateDim = 2;
  constexpr int kMeasurementDim = 1;

  // Set up measurement matrix
  Eigen::MatrixXd measurement_mat = Eigen::MatrixXd::Identity(kMeasurementDim,
                                                              kStateDim);

  // Set up measurement noise
  GaussianDistribution measurement_noise;
  measurement_noise.setDistParam(
      Eigen::VectorXd::Ones(kMeasurementDim),
      Eigen::MatrixXd::Identity(kMeasurementDim, kMeasurementDim));

  // Set up linear measurement model
  LinearMeasurementModel measurement_model(measurement_mat, measurement_noise);

  // Set up state vector for observation
  Eigen::VectorXd state = Eigen::VectorXd::Ones(kStateDim);

  // Check if observation works
  ASSERT_EQ(measurement_model.observe(state),
            Eigen::VectorXd::Ones(kMeasurementDim) * 2.0);

  // Check if dimension getters work
  ASSERT_EQ(measurement_model.getStateDim(), kStateDim);
  ASSERT_EQ(measurement_model.getMeasurementDim(), kMeasurementDim);

  // Check if Jacobian getter works
  ASSERT_EQ(measurement_model.getJacobian(), measurement_mat);

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
