#include "refill/distributions/distribution_base.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace refill {

class DistributionBaseClass : public DistributionBase<DistributionBaseClass> {
  Eigen::VectorXd mean() const override { return Eigen::VectorXd::Zero(1); }

  Eigen::MatrixXd cov() const override { return Eigen::MatrixXd::Ones(1, 1); }

  Eigen::VectorXd drawSample() override { return Eigen::VectorXd::Zero(1); }

  double evaluatePdf(const Eigen::VectorXd& x) const override { return 0.0; }
};

TEST(DistributionBaseTest, VectorizedPdfEvaluationTest) {
  DistributionBaseClass distribution;

  EXPECT_EQ(Eigen::Vector2d::Zero(2),
            distribution.evaluatePdfVectorized(Eigen::MatrixXd::Ones(2, 2)));
}

}  // namespace refill
