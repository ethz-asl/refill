#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

TEST(GaussianDistributionTest, Basics) {
  GaussianDistribution a;
  GaussianDistribution b;
  GaussianDistribution c;

  c = a + b;

  ASSERT_EQ(c.mean(), a.mean())<< "Mean not correct";
  ASSERT_EQ(c.cov(), 2.0 * a.cov())<< "Variances do not match";

  a.setDistParam(Eigen::VectorXd::Ones(2), Eigen::MatrixXd::Identity(2, 2));

  GaussianDistribution d(a);

  ASSERT_EQ(d.mean(), Eigen::VectorXd::Ones(2));

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
