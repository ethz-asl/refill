#include "refill/system_models/linear_system_model.h"

#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace refill {

TEST(LinearSystemModelTest, Fullrun) {
  const int state_dim = 2;
  const int input_dim = 1;

  Eigen::MatrixXd system_mat(state_dim, state_dim);
  Eigen::MatrixXd input_mat(state_dim, input_dim);
  GaussianDistribution<state_dim> system_noise;

  system_mat = Eigen::MatrixXd::Identity(state_dim, state_dim);
  input_mat  = Eigen::MatrixXd::Identity(state_dim, input_dim);

  system_noise.SetDistParam(Eigen::VectorXd::Ones(state_dim),
                            Eigen::MatrixXd::Identity(state_dim, state_dim));

  LinearSystemModel<state_dim, input_dim> lsm(system_mat,
                                              system_noise,
                                              input_mat);

  ASSERT_EQ(lsm.GetSystemNoise()->mean(), Eigen::VectorXd::Ones(state_dim));

  Eigen::Matrix<double, state_dim, 1> state_vec;
  Eigen::VectorXd input_vec(input_dim);

  state_vec = Eigen::VectorXd::Ones(state_dim);
  input_vec = Eigen::VectorXd::Zero(input_dim);

  lsm.Propagate(&state_vec, input_vec);

  ASSERT_EQ(state_vec, 2*Eigen::VectorXd::Ones(state_dim));
}

}  // namespace refill
