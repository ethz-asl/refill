#include "refill/system_models/linear_system_model.h"

#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace refill {

TEST(LinearSystemModelTest, Fullrun) {
  constexpr int state_dim = 2;
  constexpr int input_dim = 2;

  Eigen::MatrixXd system_mat(state_dim, state_dim);
  Eigen::MatrixXd input_mat(state_dim, input_dim);
  GaussianDistributionXd system_noise(
      Eigen::VectorXd::Zero(state_dim),
      Eigen::MatrixXd::Identity(state_dim, state_dim));

  system_mat = Eigen::MatrixXd::Identity(state_dim, state_dim);
  input_mat = Eigen::MatrixXd::Identity(state_dim, input_dim);

//  system_noise.SetDistParam(Eigen::VectorXd::Ones(state_dim),
//                            Eigen::MatrixXd::Identity(state_dim, state_dim));

  LinearSystemModelXd lsm(system_mat, system_noise, input_mat);

  ASSERT_EQ(lsm.GetSystemNoise()->mean(), Eigen::VectorXd::Zero(state_dim));

  Eigen::VectorXd state_vec(state_dim);
  Eigen::VectorXd input_vec(input_dim);

  state_vec = Eigen::VectorXd::Ones(state_dim);
  input_vec = Eigen::VectorXd::Ones(input_dim);

  state_vec = lsm.Propagate(state_vec, input_vec);

  ASSERT_EQ(2.0 * Eigen::VectorXd::Ones(state_dim), state_vec) << state_vec;
}

}  // namespace refill
