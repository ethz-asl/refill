#include "refill/measurement_models/linearized_measurement_model.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

class LinearizedMeasurementModelClass : public LinearizedMeasurementModel {
 public:
  LinearizedMeasurementModelClass(
      const size_t& state_dim, const size_t& measurement_dim,
      const DistributionInterface& measurement_noise)
      : LinearizedMeasurementModel(state_dim, measurement_dim,
                                   measurement_noise) {}
  Eigen::VectorXd observe(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& noise) const {
    return state + noise;
  }
};

TEST(LinearizedMeasurementModelTest, FullTest) {
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearizedMeasurementModelClass measurement_model(2, 2, measurement_noise);

  Eigen::MatrixXd state_jacobian = measurement_model.getMeasurementJacobian(
      Eigen::Vector2d::Zero());

  ASSERT_EQ(measurement_model.getMeasurementDim(), state_jacobian.rows());
  ASSERT_EQ(measurement_model.getStateDim(), state_jacobian.cols());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), state_jacobian);

  Eigen::Matrix2d noise_jacobian = measurement_model.getNoiseJacobian(
      Eigen::Vector2d::Zero());

  ASSERT_EQ(measurement_model.getMeasurementDim(), noise_jacobian.rows());
  ASSERT_EQ(measurement_model.getNoiseDim(), noise_jacobian.cols());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), noise_jacobian);
}

}  // namespace refill
