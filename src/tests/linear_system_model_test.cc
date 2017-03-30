#include "refill/system_models/linear_system_model.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

TEST(LinearSystemModelTest, ConstructorTest) {
  LinearSystemModel system_model_1;

  EXPECT_EQ(0, system_model_1.getStateDim());
  EXPECT_EQ(0, system_model_1.getInputDim());
  EXPECT_EQ(0, system_model_1.getNoiseDim());

  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearSystemModel system_model_2(Eigen::Matrix2d::Identity(), system_noise);

  EXPECT_EQ(2, system_model_2.getStateDim());
  EXPECT_EQ(0, system_model_2.getInputDim());
  EXPECT_EQ(2, system_model_2.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_2.getSystemMapping());
  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), system_model_2.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_2.getNoiseMapping());

  LinearSystemModel system_model_3(Eigen::Matrix2d::Identity(), system_noise,
                                   Eigen::Matrix2d::Identity());

  EXPECT_EQ(2, system_model_3.getStateDim());
  EXPECT_EQ(2, system_model_3.getInputDim());
  EXPECT_EQ(2, system_model_3.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_3.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_3.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_3.getNoiseMapping());

  LinearSystemModel system_model_4(Eigen::Matrix2d::Identity(), system_noise,
                                   Eigen::Matrix2d::Identity(),
                                   Eigen::Matrix2d::Ones());

  EXPECT_EQ(2, system_model_4.getStateDim());
  EXPECT_EQ(2, system_model_4.getInputDim());
  EXPECT_EQ(2, system_model_4.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_4.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_4.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Ones(), system_model_4.getNoiseMapping());
}

TEST(LinearSystemModelTest, SetterTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearSystemModel system_model;

  system_model.setSystemParameters(Eigen::Matrix2d::Identity(), system_noise);

  EXPECT_EQ(2, system_model.getStateDim());
  EXPECT_EQ(0, system_model.getInputDim());
  EXPECT_EQ(2, system_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model.getNoiseMapping());

  system_noise.setDistributionParameters(Eigen::Vector3d::Zero(),
                                         Eigen::Matrix3d::Identity());

  system_model.setSystemParameters(Eigen::Matrix3d::Identity(), system_noise,
                                   Eigen::Matrix3d::Identity());

  EXPECT_EQ(3, system_model.getStateDim());
  EXPECT_EQ(3, system_model.getInputDim());
  EXPECT_EQ(3, system_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getNoiseMapping());

  system_noise.setDistributionParameters(Eigen::Vector4d::Zero(),
                                         Eigen::Matrix4d::Identity());

  system_model.setSystemParameters(Eigen::Matrix4d::Identity(), system_noise,
                                   Eigen::Matrix4d::Identity(),
                                   Eigen::Matrix4d::Ones());

  EXPECT_EQ(4, system_model.getStateDim());
  EXPECT_EQ(4, system_model.getInputDim());
  EXPECT_EQ(4, system_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix4d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix4d::Identity(), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix4d::Ones(), system_model.getNoiseMapping());
}

TEST(LinearSystemModelTest, GetterTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearSystemModel system_model(Eigen::Matrix2d::Identity(), system_noise,
                                 Eigen::Matrix2d::Ones(),
                                 Eigen::Matrix2d::Constant(2.0));

  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix2d::Ones(), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Constant(2.0), system_model.getNoiseMapping());
  EXPECT_EQ(
      Eigen::Matrix2d::Identity(),
      system_model.getStateJacobian(Eigen::Vector2d::Zero(),
                                    Eigen::Vector2d::Zero()));
  EXPECT_EQ(
      Eigen::Matrix2d::Constant(2.0),
      system_model.getNoiseJacobian(Eigen::Vector2d::Zero(),
                                    Eigen::Vector2d::Zero()));
}

TEST(LinearSystemModelTest, PropagationTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Ones(),
                                    Eigen::Matrix2d::Identity());

  LinearSystemModel system_model(Eigen::Matrix2d::Identity(), system_noise,
                                 Eigen::Matrix2d::Identity(),
                                 Eigen::Matrix2d::Ones());

  EXPECT_EQ(Eigen::Vector2d::Constant(3.0),
            system_model.propagate(Eigen::Vector2d::Ones()));
  EXPECT_EQ(
      Eigen::Vector2d::Constant(4.0),
      system_model.propagate(Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones(),
                             Eigen::Vector2d::Ones()));
}

}  // namespace refill
