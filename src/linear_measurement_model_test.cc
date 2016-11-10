#include "refill/measurement_models/linear_measurement_model.h"

#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace refill {

TEST(LinearMeasurementModelTest, Fullrun) {
  constexpr int state_dim = 2;
  constexpr int meas_dim = 1;

  Eigen::MatrixXd measurement_mat(meas_dim, state_dim);

  measurement_mat = Eigen::MatrixXd::Identity(meas_dim, state_dim);
  GaussianDistribution<meas_dim> measurement_noise;

  measurement_noise.SetDistParam(Eigen::VectorXd::Zero(meas_dim),
                                 Eigen::MatrixXd::Identity(meas_dim, meas_dim));

  LinearMeasurementModel<state_dim, meas_dim> lmm(measurement_mat,
                                                  measurement_noise);

  Eigen::VectorXd state(state_dim);

  state = Eigen::VectorXd::Ones(state_dim);

  ASSERT_EQ(lmm.Observe(state), Eigen::VectorXd::Ones(meas_dim));
}

}  // namespace refill
