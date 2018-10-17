#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"

namespace refill {

TEST(GaussianDistributionTest, ConstructorTests) {
  GaussianDistribution* dist = new GaussianDistribution();

  ASSERT_EQ(1, dist->dimension())<< "Dimension not correct.";
  ASSERT_EQ(Eigen::VectorXd::Zero(1), dist->mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::MatrixXd::Identity(1, 1), dist->cov())
      << "Covariance not correct";

  delete dist;
  dist = new GaussianDistribution(2);

  ASSERT_EQ(2, dist->dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Zero(), dist->mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity(), dist->cov())
      << "Covariance not correct";

  delete dist;
  dist = new GaussianDistribution(Eigen::Vector2d::Zero(),
                                  Eigen::Matrix2d::Identity());

  ASSERT_EQ(2, dist->dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Zero(), dist->mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity(), dist->cov())
      << "Covariance not correct";

  GaussianDistribution dist_4(*dist);

  ASSERT_EQ(2, dist_4.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Zero(), dist_4.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity(), dist_4.cov())
      << "Covariance not correct";
}

TEST(GaussianDistributionTest, SetterTests) {
  GaussianDistribution dist(2);

  dist.setDistributionParameters(Eigen::Vector2d::Ones(),
                                 Eigen::Matrix2d::Identity() * 2.0);

  ASSERT_EQ(2, dist.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Ones(), dist.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 2.0, dist.cov())
      << "Covariance not correct";

  dist.setMean(Eigen::Vector2d::Constant(2.0));

  ASSERT_EQ(2, dist.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Constant(2.0), dist.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 2.0, dist.cov())
      << "Covariance not correct";

  dist.setCov(Eigen::Matrix2d::Identity() * 3.0);

  ASSERT_EQ(2, dist.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Constant(2.0), dist.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 3.0, dist.cov())
      << "Covariance not correct";
}

TEST(GaussianDistributionTest, OperatorTests) {
  GaussianDistribution dist_1(Eigen::Vector2d::Zero(),
                              Eigen::Matrix2d::Identity());
  GaussianDistribution dist_2(Eigen::Vector2d::Ones(),
                              Eigen::Matrix2d::Identity() * 2.0);

  dist_1 += dist_2;

  ASSERT_EQ(2, dist_1.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Ones(), dist_1.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 3.0, dist_1.cov())
      << "Covariance not correct";

  dist_1 -= dist_2;

  ASSERT_EQ(2, dist_1.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Zero(), dist_1.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 5.0, dist_1.cov())
      << "Covariance not correct";

  GaussianDistribution dist_3 = dist_1 + dist_2;

  ASSERT_EQ(2, dist_3.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Ones(), dist_3.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 7.0, dist_3.cov())
      << "Covariance not correct";

  GaussianDistribution dist_4 = dist_1 - dist_2;

  ASSERT_EQ(2, dist_4.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Constant(-1.0), dist_4.mean())
      << "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 7.0, dist_4.cov())
      << "Covariance not correct";

  dist_4.setDistributionParameters(Eigen::Vector2d::Ones(),
                                   Eigen::Matrix2d::Identity());

  GaussianDistribution dist_5 = Eigen::Matrix2d::Ones() * dist_4;

  ASSERT_EQ(2, dist_5.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Constant(2.0), dist_5.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Constant(2.0), dist_5.cov())
      << "Covariance not correct";

  GaussianDistribution dist_6 = 2.0 * dist_4;

  ASSERT_EQ(2, dist_6.dimension())<< "Dimension not correct";
  ASSERT_EQ(Eigen::Vector2d::Constant(2.0), dist_6.mean())<< "Mean not correct";
  ASSERT_EQ(Eigen::Matrix2d::Identity() * 4.0, dist_6.cov())
      <<"Covariance not correct";
}

}  // namespace refill
