#include "refill/system_models/linear_system_model.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

TEST(LinearSystemModelTest, ConstructorTest) {
  // test default constructor
  LinearSystemModel* system_model_p = new LinearSystemModel;

  EXPECT_EQ(0, system_model_p->getStateDim());
  EXPECT_EQ(0, system_model_p->getInputDim());
  EXPECT_EQ(0, system_model_p->getSystemNoiseDim());

  delete system_model_p;

  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  // test constructor with system mapping and noise
  system_model_p = new LinearSystemModel(Eigen::Matrix2d::Identity(),
                                         system_noise);

  EXPECT_EQ(2, system_model_p->getStateDim());
  EXPECT_EQ(0, system_model_p->getInputDim());
  EXPECT_EQ(2, system_model_p->getSystemNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getSystemMapping());
  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), system_model_p->getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getNoiseMapping());

  delete system_model_p;

  // test constructor with system mapping, noise and input mapping
  system_model_p = new LinearSystemModel(Eigen::Matrix2d::Identity(),
                                         system_noise,
                                         Eigen::Matrix2d::Identity());

  EXPECT_EQ(2, system_model_p->getStateDim());
  EXPECT_EQ(2, system_model_p->getInputDim());
  EXPECT_EQ(2, system_model_p->getSystemNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getSystemMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getNoiseMapping());

  delete system_model_p;

  // test constructor with system noise and system, input and noise mapping
  system_model_p = new LinearSystemModel(Eigen::Matrix2d::Identity(),
                                         system_noise,
                                         Eigen::Matrix2d::Identity(),
                                         Eigen::Matrix2d::Ones());

  EXPECT_EQ(2, system_model_p->getStateDim());
  EXPECT_EQ(2, system_model_p->getInputDim());
  EXPECT_EQ(2, system_model_p->getSystemNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getSystemMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model_p->getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Ones(), system_model_p->getNoiseMapping());

  delete system_model_p;
}

TEST(LinearSystemModelTest, SetterTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearSystemModel system_model;

  system_model.setModelParameters(Eigen::Matrix2d::Identity(), system_noise);

  EXPECT_EQ(2, system_model.getStateDim());
  EXPECT_EQ(0, system_model.getInputDim());
  EXPECT_EQ(2, system_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), system_model.getNoiseMapping());

  system_noise.setDistributionParameters(Eigen::Vector3d::Zero(),
                                         Eigen::Matrix3d::Identity());

  system_model.setModelParameters(Eigen::Matrix3d::Identity(), system_noise,
                                   Eigen::Matrix3d::Identity());

  EXPECT_EQ(3, system_model.getStateDim());
  EXPECT_EQ(3, system_model.getInputDim());
  EXPECT_EQ(3, system_model.getNoiseDim());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getSystemMapping());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getInputMapping());
  EXPECT_EQ(Eigen::Matrix3d::Identity(), system_model.getNoiseMapping());

  system_noise.setDistributionParameters(Eigen::Vector4d::Zero(),
                                         Eigen::Matrix4d::Identity());

  system_model.setModelParameters(Eigen::Matrix4d::Identity(), system_noise,
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
