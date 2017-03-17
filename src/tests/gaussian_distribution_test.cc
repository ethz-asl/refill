#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

TEST(GaussianDistributionTest, Basics) {
  GaussianDistribution a;
  GaussianDistribution b;
  GaussianDistribution c;

  c = a + b;

  ASSERT_EQ(a.mean(), c.mean())<< "Mean not correct";
  ASSERT_EQ(a.cov() * 2.0, c.cov())<< "Variances do not match";

  a.setDistParam(Eigen::VectorXd::Ones(2), Eigen::MatrixXd::Identity(2, 2));

  GaussianDistribution d(a);
  ASSERT_EQ(Eigen::VectorXd::Ones(2), d.mean());
  ASSERT_EQ(Eigen::MatrixXd::Identity(2, 2), d.cov());

  d.setMean(Eigen::VectorXd::Zero(2));
  ASSERT_EQ(Eigen::VectorXd::Zero(2), d.mean());

  d.setCov(Eigen::MatrixXd::Identity(2, 2) * 2.0);
  ASSERT_EQ(Eigen::MatrixXd::Identity(2, 2) * 2.0, d.cov());

  GaussianDistribution e = Eigen::MatrixXd::Constant(2, 2, 2.0) * a;

  ASSERT_EQ(Eigen::VectorXd::Ones(2) * 2.0, e.mean());
  ASSERT_EQ(Eigen::MatrixXd::Constant(2, 2, 4.0), e.cov());

  GaussianDistribution f = 2.0 * a;

  ASSERT_EQ(Eigen::VectorXd::Ones(2) * 2.0, f.mean());
  ASSERT_EQ(Eigen::MatrixXd::Constant(2, 2, 4.0), f.cov());

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
