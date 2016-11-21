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

  // Test distribution parameter setters
  a.setDistParam(Eigen::VectorXd::Ones(2), Eigen::MatrixXd::Identity(2, 2));

  ASSERT_EQ(a.mean(), Eigen::VectorXd::Ones(2));
  ASSERT_EQ(a.cov(), Eigen::MatrixXd::Identity(2, 2));

  b.setMean(Eigen::VectorXd::Ones(1));
  b.setCov(Eigen::MatrixXd::Identity(1, 1) * 2.0);

  ASSERT_EQ(b.mean(), Eigen::VectorXd::Ones(1));
  ASSERT_EQ(b.cov(), Eigen::MatrixXd::Identity(1, 1) * 2.0);

  // Test copy constructor
  GaussianDistribution d(a);
  ASSERT_EQ(d.mean(), Eigen::VectorXd::Ones(2));

  // Test dimension constructor
  GaussianDistribution dim_test(5);
  ASSERT_EQ(dim_test.mean().size(), 5);

  // Test dimension function
  ASSERT_EQ(dim_test.dimension(), 5);

  // Test constructor with mean and covariance
  GaussianDistribution mean_cov_test(Eigen::VectorXd::Ones(3),
                                     Eigen::MatrixXd::Identity(3, 3));
  ASSERT_EQ(mean_cov_test.mean(), Eigen::VectorXd::Ones(3));
  ASSERT_EQ(mean_cov_test.cov(), Eigen::MatrixXd::Identity(3, 3));

  // Test

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
