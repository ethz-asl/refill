#include "refill/system_models/system_model_base.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

class SystemModelClass : public SystemModelBase {
 public:
  SystemModelClass(const size_t& state_dim,
                   const GaussianDistribution& system_noise)
      : SystemModelBase(state_dim, system_noise) {
  }
  SystemModelClass(const size_t& state_dim,
                   const GaussianDistribution& system_noise,
                   const size_t& input_dim)
      : SystemModelBase(state_dim, system_noise, input_dim) {
  }
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input,
                            const Eigen::VectorXd& noise) const {
    return state + input + noise;
  }
  void setModelParameters(const size_t& state_dim,
                          const DistributionInterface& system_noise) {
    this->setSystemModelBaseParameters(state_dim, system_noise);
  }
  void setModelParameters(const size_t& state_dim,
                          const DistributionInterface& system_noise,
                          const size_t& input_dim) {
    this->setSystemModelBaseParameters(state_dim, system_noise, input_dim);
  }
};

TEST(SystemModelBaseTest, NoInputTest) {
  GaussianDistribution system_noise(Eigen::Vector3d::Zero(),
                                    Eigen::Matrix3d::Identity());

  // Test constructor with two parameters
  SystemModelClass system_model(3, system_noise);

  ASSERT_EQ(3, system_model.getStateDim());
  ASSERT_EQ(0, system_model.getInputDim());
  ASSERT_EQ(3, system_model.getNoiseDim());
  ASSERT_EQ(Eigen::Vector3d::Zero(), system_model.getNoise()->mean());
  ASSERT_EQ(Eigen::Matrix3d::Identity(), system_model.getNoise()->cov());

  system_noise.setDistributionParameters(Eigen::Vector2d::Ones(),
                                         Eigen::Matrix2d::Identity() * 2.0);

  system_model.setModelParameters(2, system_noise);

  ASSERT_EQ(2, system_model.getStateDim());
  ASSERT_EQ(0, system_model.getInputDim());
  ASSERT_EQ(2, system_model.getNoiseDim());
  ASSERT_EQ(Eigen::Vector2d::Ones(), system_model.getNoise()->mean());
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 2.0,
            system_model.getNoise()->cov());

  Eigen::Matrix2d state_samples = Eigen::Matrix2d::Ones();
  Eigen::Matrix2d noise_samples = Eigen::Matrix2d::Ones();
  Eigen::Vector2d input = Eigen::Vector2d::Ones();

  Eigen::MatrixXd vectorized_propagation = system_model.propagateVectorized(
      state_samples, input, noise_samples);

  ASSERT_EQ(2, vectorized_propagation.rows());
  ASSERT_EQ(4, vectorized_propagation.cols());
  ASSERT_EQ(Eigen::MatrixXd::Ones(2, 4) * 3.0, vectorized_propagation);
}

TEST(SystemModelBaseTest, WithInputTest) {
  GaussianDistribution system_noise(Eigen::Vector3d::Zero(),
                                    Eigen::Matrix3d::Identity());

  // Test constructor with two parameters
  SystemModelClass system_model(3, system_noise, 3);

  ASSERT_EQ(3, system_model.getStateDim());
  ASSERT_EQ(3, system_model.getInputDim());
  ASSERT_EQ(3, system_model.getNoiseDim());
  ASSERT_EQ(Eigen::Vector3d::Zero(), system_model.getNoise()->mean());
  ASSERT_EQ(Eigen::Matrix3d::Identity(), system_model.getNoise()->cov());

  system_noise.setDistributionParameters(Eigen::Vector2d::Ones(),
                                         Eigen::Matrix2d::Identity() * 2.0);

  system_model.setModelParameters(2, system_noise, 2);

  ASSERT_EQ(2, system_model.getStateDim());
  ASSERT_EQ(2, system_model.getInputDim());
  ASSERT_EQ(2, system_model.getNoiseDim());
  ASSERT_EQ(Eigen::Vector2d::Ones(), system_model.getNoise()->mean());
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 2.0,
            system_model.getNoise()->cov());

  Eigen::Matrix2d state_samples = Eigen::Matrix2d::Ones();
  Eigen::Matrix2d noise_samples = Eigen::Matrix2d::Ones();
  Eigen::Vector2d input = Eigen::Vector2d::Ones();

  Eigen::MatrixXd vectorized_propagation = system_model.propagateVectorized(
      state_samples, input, noise_samples);

  ASSERT_EQ(2, vectorized_propagation.rows());
  ASSERT_EQ(4, vectorized_propagation.cols());
  ASSERT_EQ(Eigen::MatrixXd::Ones(2, 4) * 3.0, vectorized_propagation);
}

}  // namespace refill
