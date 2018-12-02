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

  Eigen::Matrix2d sampled_state = Eigen::Matrix2d::Zero();
  Eigen::Vector2d measurement = Eigen::Vector2d::Zero();

  Eigen::VectorXd likelihoods =
      likelihood.getLikelihoodVectorized(sampled_state, measurement);

  EXPECT_EQ(Eigen::Vector2d::Zero(), likelihoods);
}

}  // namespace refill
