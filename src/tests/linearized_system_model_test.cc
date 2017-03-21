#include "refill/system_models/linearized_system_model.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

class LinearizedSystemModelClass : public LinearizedSystemModel {
 public:
  LinearizedSystemModelClass(const size_t& state_dim,
                             const DistributionInterface& system_noise)
      : LinearizedSystemModel(state_dim, system_noise) {}
  LinearizedSystemModelClass(const size_t& state_dim,
                             const DistributionInterface& system_noise,
                             const size_t& input_dim)
      : LinearizedSystemModel(state_dim, system_noise, input_dim) {}
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input,
                            const Eigen::VectorXd& noise) const {
    return state + noise;
  }
};

TEST(LinearizedSystemModelTest, NoInputTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearizedSystemModelClass system_model(2, system_noise);

  Eigen::MatrixXd state_jacobian = system_model.getStateJacobian(
      Eigen::Vector2d::Zero(), Eigen::VectorXd::Zero(0));

  ASSERT_EQ(state_jacobian.rows(), state_jacobian.cols());
  ASSERT_EQ(system_model.getStateDim(), state_jacobian.rows());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), state_jacobian);

  Eigen::MatrixXd noise_jacobian = system_model.getNoiseJacobian(
      Eigen::Vector2d::Zero(), Eigen::VectorXd::Zero(0));

  ASSERT_EQ(system_model.getStateDim(), noise_jacobian.rows());
  ASSERT_EQ(system_model.getSystemNoiseDim(), noise_jacobian.cols());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), noise_jacobian);
}

TEST(LinearizedSystemModelTest, WithInputTest) {
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());

  LinearizedSystemModelClass system_model(2, system_noise, 2);

  Eigen::MatrixXd state_jacobian = system_model.getStateJacobian(
      Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

  ASSERT_EQ(state_jacobian.rows(), state_jacobian.cols());
  ASSERT_EQ(system_model.getStateDim(), state_jacobian.rows());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), state_jacobian);

  Eigen::MatrixXd noise_jacobian = system_model.getNoiseJacobian(
      Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

  ASSERT_EQ(system_model.getStateDim(), noise_jacobian.rows());
  ASSERT_EQ(system_model.getSystemNoiseDim(), noise_jacobian.cols());
  ASSERT_EQ(Eigen::Matrix2d::Identity(), noise_jacobian);
}

}  // namespace refill
