#include "refill/measurement_models/linear_measurement_model.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(LinearMeasurementModelTest, Fullrun) {
  constexpr int kStateDim = 2;
  constexpr int kMeasurementDim = 1;

  Eigen::MatrixXd measurement_mat(kMeasurementDim, kStateDim);
  measurement_mat = Eigen::MatrixXd::Identity(kMeasurementDim, kStateDim);

  GaussianDistribution<kMeasurementDim> measurement_noise;
  measurement_noise.setDistParam(
      Eigen::VectorXd::Ones(kMeasurementDim),
      Eigen::MatrixXd::Identity(kMeasurementDim, kMeasurementDim));

  LinearMeasurementModel<kStateDim, kMeasurementDim> linear_measurement_model(
      measurement_mat, measurement_noise);

  Eigen::VectorXd state(kStateDim);

  state = Eigen::VectorXd::Ones(kStateDim);

  ASSERT_EQ(linear_measurement_model.observe(state),
            Eigen::VectorXd::Ones(kMeasurementDim) * 2.0);
}

}  // namespace refill
