#include "refill/distributions/likelihood.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

class LikelihoodClass : public Likelihood {
  double getLikelihood(const Eigen::VectorXd& state,
                       const Eigen::VectorXd& measurement) const override {
    return 0;
  }
};

TEST(LikelihoodTest, GetLikelihoodVectorizedTest) {
  LikelihoodClass likelihood;

  Eigen::MatrixXd sampled_state = Eigen::MatrixXd::Zero(2, 2);
  Eigen::VectorXd measurement = Eigen::VectorXd::Zero(2);

  Eigen::VectorXd likelihoods =
      likelihood.getLikelihoodVectorized(sampled_state, measurement);

  EXPECT_EQ(Eigen::Vector2d::Zero(), likelihoods);
}

}  // namespace refill
