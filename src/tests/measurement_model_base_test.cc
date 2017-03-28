#include "refill/measurement_models/measurement_model_base.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

class MeasurementModelClass : public MeasurementModelBase {
 public:
  MeasurementModelClass(const size_t& state_dim, const size_t& measurement_dim,
                        const DistributionInterface& measurement_noise)
      : MeasurementModelBase(state_dim, measurement_dim, measurement_noise) {
  }

  void setModelParameters(const size_t& state_dim,
                          const size_t& measurement_dim,
                          const DistributionInterface& measurement_noise) {
    this->setMeasurementModelBaseParameters(state_dim, measurement_dim,
                                            measurement_noise);
  }
  Eigen::VectorXd observe(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& noise) const {
    return state + noise;
  }
};

TEST(MeasurementModelBaseTest, FullTest) {
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  MeasurementModelClass measurement_model(2, 2, measurement_noise);

  ASSERT_EQ(2, measurement_model.getStateDim());
  ASSERT_EQ(2, measurement_model.getMeasurementDim());
  ASSERT_EQ(2, measurement_model.getMeasurementNoiseDim());
  ASSERT_EQ(Eigen::Vector2d::Zero(),
            measurement_model.getMeasurementNoise()->mean());
  ASSERT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model.getMeasurementNoise()->cov());

  measurement_noise.setDistributionParameters(
      Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity() * 2.0);

  measurement_model.setModelParameters(3, 3, measurement_noise);

  ASSERT_EQ(3, measurement_model.getStateDim());
  ASSERT_EQ(3, measurement_model.getMeasurementDim());
  ASSERT_EQ(3, measurement_model.getMeasurementNoiseDim());
  ASSERT_EQ(Eigen::Vector3d::Ones(),
            measurement_model.getMeasurementNoise()->mean());
  ASSERT_EQ(Eigen::Matrix3d::Identity() * 2.0,
            measurement_model.getMeasurementNoise()->cov());
}

}  // namespace refill
