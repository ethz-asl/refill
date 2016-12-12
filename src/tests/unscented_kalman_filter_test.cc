#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/unscented_kalman_filter.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/system_models/linear_system_model.h"

namespace refill {

TEST(UnscentedKalmanFilterTest, ElementaryTest) {
  // Initializes default 1d models
  GaussianDistribution initial_state;
  GaussianDistribution system_noise;
  GaussianDistribution measurement_noise;
  LinearSystemModel system_model(Eigen::MatrixXd::Identity(1, 1), system_noise);
  LinearMeasurementModel measurement_model(Eigen::MatrixXd::Identity(1, 1),
                                           measurement_noise);

  UnscentedKalmanFilter unscented_kf(initial_state);

  unscented_kf.predict(system_model);
  unscented_kf.update(measurement_model, Eigen::VectorXd::Zero(1));
}

}  // namespace refill
