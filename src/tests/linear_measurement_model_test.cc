#include "refill/measurement_models/linear_measurement_model.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(LinearMeasurementModelTest, ConstructorTest) {
  LinearMeasurementModel measurement_model_1;

  EXPECT_EQ(0, measurement_model_1.getStateDim());
  EXPECT_EQ(0, measurement_model_1.getMeasurementDim());
  EXPECT_EQ(0, measurement_model_1.getNoiseDim());

  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearMeasurementModel measurement_model_2(Eigen::Matrix2d::Identity(),
                                             measurement_noise);

  EXPECT_EQ(2, measurement_model_2.getStateDim());
  EXPECT_EQ(2, measurement_model_2.getMeasurementDim());
  EXPECT_EQ(2, measurement_model_2.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model_2.getMeasurementMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), measurement_model_2.getNoiseMapping());

  LinearMeasurementModel measurement_model_3(Eigen::Matrix2d::Identity(),
                                             measurement_noise,
                                             Eigen::Matrix2d::Ones());

  EXPECT_EQ(2, measurement_model_3.getStateDim());
  EXPECT_EQ(2, measurement_model_3.getMeasurementDim());
  EXPECT_EQ(2, measurement_model_3.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model_3.getMeasurementMapping());
  EXPECT_EQ(Eigen::Matrix2d::Ones(), measurement_model_3.getNoiseMapping());
}

TEST(LinearMeasurementModelTest, SetterTest) {
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearMeasurementModel measurement_model;

  measurement_model.setMeasurementParameters(Eigen::Matrix2d::Identity(),
                                             measurement_noise);

  EXPECT_EQ(2, measurement_model.getStateDim());
  EXPECT_EQ(2, measurement_model.getMeasurementDim());
  EXPECT_EQ(2, measurement_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model.getMeasurementMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), measurement_model.getNoiseMapping());

  measurement_noise.setDistributionParameters(Eigen::Vector3d::Zero(),
                                              Eigen::Matrix3d::Identity());

  measurement_model.setMeasurementParameters(Eigen::Matrix3d::Identity(),
                                             measurement_noise,
                                             Eigen::Matrix3d::Ones());

  EXPECT_EQ(3, measurement_model.getStateDim());
  EXPECT_EQ(3, measurement_model.getMeasurementDim());
  EXPECT_EQ(3, measurement_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix3d::Identity(),
            measurement_model.getMeasurementMapping());
  EXPECT_EQ(Eigen::Matrix3d::Ones(), measurement_model.getNoiseMapping());
}

TEST(LinearMeasurementModelTest, GetterTest) {
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearMeasurementModel measurement_model(Eigen::Matrix2d::Identity(),
                                           measurement_noise,
                                           Eigen::Matrix2d::Ones());

  EXPECT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model.getMeasurementMapping());
  EXPECT_EQ(Eigen::Matrix2d::Ones(), measurement_model.getNoiseMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(),
            measurement_model.getMeasurementJacobian(Eigen::Vector2d::Zero()));
  EXPECT_EQ(Eigen::Matrix2d::Ones(),
            measurement_model.getNoiseJacobian(Eigen::Vector2d::Zero()));
}

TEST(LinearMeasurementModel, ObservationTest) {
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearMeasurementModel measurement_model(Eigen::Matrix2d::Identity(),
                                           measurement_noise,
                                           Eigen::Matrix2d::Ones());

  EXPECT_EQ(
      Eigen::Vector2d::Constant(3.0),
      measurement_model.observe(Eigen::Vector2d::Ones(),
                                Eigen::Vector2d::Ones()));
}

}  // namespace refill
