#include "refill/system_models/linear_system_model.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(LinearSystemModelTest, Fullrun) {
  constexpr int kStateDim = 2;
  constexpr int kInputDim = 1;

  Eigen::MatrixXd system_mat(kStateDim, kStateDim);
  Eigen::MatrixXd input_mat(kStateDim, kInputDim);
  GaussianDistribution<kStateDim> system_noise;

  system_mat = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
  input_mat = Eigen::MatrixXd::Identity(kStateDim, kInputDim);

  system_noise.setDistParam(Eigen::VectorXd::Ones(kStateDim),
                            Eigen::MatrixXd::Identity(kStateDim, kStateDim));

  LinearSystemModel<kStateDim, kInputDim> linear_system_model(system_mat,
                                                              system_noise,
                                                              input_mat);

  ASSERT_EQ(linear_system_model.getSystemNoise()->mean(),
            Eigen::VectorXd::Ones(kStateDim));

  Eigen::Matrix<double, kStateDim, 1> state_vec;
  Eigen::VectorXd input_vec(kInputDim);

  state_vec = Eigen::VectorXd::Ones(kStateDim);
  input_vec = Eigen::VectorXd::Zero(kInputDim);

  linear_system_model.propagate(&state_vec, input_vec);

  ASSERT_EQ(state_vec, 2 * Eigen::VectorXd::Ones(kStateDim));
}

}  // namespace refill
