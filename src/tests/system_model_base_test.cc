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

TEST(SystemModelBaseTest, Fullrun) {
  constexpr size_t kStateDim = 2;
  constexpr size_t kInputDim = 2;
  constexpr size_t kNoiseDim = 2;

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(kNoiseDim);
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(kNoiseDim, kNoiseDim);
  GaussianDistribution system_noise(mean, covariance);

  // Test constructor with two parameters
  SystemModelClass system_model_1(kStateDim, system_noise);

  ASSERT_EQ(system_model_1.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_1.getInputDim(), 0);
  ASSERT_EQ(system_model_1.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_1.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_1.getSystemNoise()->cov(), covariance);

  system_model_1.setModelParameters(kStateDim, system_noise);

  ASSERT_EQ(system_model_1.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_1.getInputDim(), 0);
  ASSERT_EQ(system_model_1.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_1.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_1.getSystemNoise()->cov(), covariance);

  Eigen::MatrixXd state_samples = Eigen::MatrixXd::Ones(kStateDim, 2);
  Eigen::MatrixXd noise_samples = Eigen::MatrixXd::Ones(kNoiseDim, 2);
  Eigen::VectorXd input = Eigen::VectorXd::Ones(kInputDim);

  Eigen::MatrixXd vectorized_propagation = system_model_1.propagateVectorized(
      state_samples, input, noise_samples);

  ASSERT_EQ(kStateDim, vectorized_propagation.rows());
  ASSERT_EQ(4, vectorized_propagation.cols());
  ASSERT_EQ(Eigen::MatrixXd::Ones(kStateDim, 4) * 3.0, vectorized_propagation);

  // Test constructor with three parameters
  SystemModelClass system_model_2(kStateDim, system_noise, kInputDim);

  ASSERT_EQ(system_model_2.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_2.getInputDim(), kInputDim);
  ASSERT_EQ(system_model_2.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_2.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_2.getSystemNoise()->cov(), covariance);

  system_model_2.setModelParameters(kStateDim, system_noise, kInputDim);

  ASSERT_EQ(system_model_2.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_2.getInputDim(), kInputDim);
  ASSERT_EQ(system_model_2.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_2.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_2.getSystemNoise()->cov(), covariance);
}

}  // namespace refill
