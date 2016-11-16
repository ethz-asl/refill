#include "refill/system_models/linear_system_model.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace refill {

TEST(LinearSystemModelTest, Fullrun) {
  constexpr int kStateDim = 2;
  constexpr int kInputDim = 2;

  // Set up system and input matrix
  Eigen::MatrixXd system_mat = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
  Eigen::MatrixXd input_mat = Eigen::MatrixXd::Identity(kStateDim, kInputDim);

  // Set up system noise, with non standard params
  GaussianDistribution system_noise;
  system_noise.setDistParam(Eigen::VectorXd::Ones(kStateDim),
                            Eigen::MatrixXd::Identity(kStateDim, kStateDim));

  // Set up system model
  LinearSystemModel system_model(system_mat, system_noise, input_mat);

  // Set up state and input vector
  Eigen::VectorXd state_vec = Eigen::VectorXd::Ones(kStateDim);
  Eigen::VectorXd input_vec = Eigen::VectorXd::Ones(kInputDim);

  // Propagate the state vector through the system
  state_vec = system_model.propagate(state_vec, input_vec);

  // Check that propagation was correct
  ASSERT_EQ(state_vec, Eigen::VectorXd::Ones(kStateDim) * 3.0);

  // Check that dimension getters work
  ASSERT_EQ(system_model.getStateDim(), kStateDim);
  ASSERT_EQ(system_model.getInputDim(), kInputDim);

  // Check that Jacobian getter works
  ASSERT_EQ(system_model.getJacobian(), system_mat);

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
