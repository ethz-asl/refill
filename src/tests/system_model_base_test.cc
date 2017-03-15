#include "refill/system_models/system_model_base.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

TEST(SystemModelBaseTest, Fullrun) {
  constexpr size_t kStateDim = 2;
  constexpr size_t kInputDim = 2;
  constexpr size_t kNoiseDim = 2;

  class SystemModelClass : SystemModelBase {
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
  };

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(kNoiseDim);
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(kNoiseDim, kNoiseDim);
  GaussianDistribution system_noise(mean, covariance);

  SystemModelClass system_model_1(kStateDim, system_noise);

  ASSERT_EQ(system_model_1.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_1.getInputDim(), 0);
  ASSERT_EQ(system_model_1.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_1.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_1.getSystemNoise()->cov(), covariance);

  Eigen::VectorXd propagated = system_model_1.propagate(
      Eigen::VectorXd::Ones(kStateDim), Eigen::VectorXd::Ones(kInputDim),
      Eigen::VectorXd::Ones(kNoiseDim));

  ASSERT_EQ(propagated, Eigen::VectorXd::Ones(kStateDim) * 3.0);

  system_model_1.setSystemModelBaseParameters(kStateDim, system_noise);

  ASSERT_EQ(system_model_1.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_1.getInputDim(), 0);
  ASSERT_EQ(system_model_1.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_1.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_1.getSystemNoise()->cov(), covariance);

  propagated = system_model_1.propagate(Eigen::VectorXd::Ones(kStateDim),
                                        Eigen::VectorXd::Ones(kInputDim),
                                        Eigen::VectorXd::Ones(kNoiseDim));

  ASSERT_EQ(propagated, Eigen::VectorXd::Ones(kStateDim) * 3.0);

  SystemModelClass system_model_2(kStateDim, system_noise, kInputDim);

  ASSERT_EQ(system_model_2.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_2.getInputDim(), kInputDim);
  ASSERT_EQ(system_model_2.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_2.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_2.getSystemNoise()->cov(), covariance);

  propagated = system_model_2.propagate(Eigen::VectorXd::Ones(kStateDim),
                                        Eigen::VectorXd::Ones(kInputDim),
                                        Eigen::VectorXd::Ones(kNoiseDim));

  ASSERT_EQ(propagated, Eigen::VectorXd::Ones(kStateDim) * 3.0);

  system_model_2.setSystemModelBaseParameters(kStateDim, system_noise,
                                              kInputDim);

  ASSERT_EQ(system_model_2.getStateDim(), kStateDim);
  ASSERT_EQ(system_model_2.getInputDim(), kInputDim);
  ASSERT_EQ(system_model_2.getSystemNoiseDim(), kNoiseDim);
  ASSERT_EQ(system_model_2.getSystemNoise()->mean(), mean);
  ASSERT_EQ(system_model_2.getSystemNoise()->cov(), covariance);

  propagated = system_model_2.propagate(Eigen::VectorXd::Ones(kStateDim),
                                        Eigen::VectorXd::Ones(kInputDim),
                                        Eigen::VectorXd::Ones(kNoiseDim));

  ASSERT_EQ(propagated, Eigen::VectorXd::Ones(kStateDim) * 3.0);
}

}  // namespace refill
