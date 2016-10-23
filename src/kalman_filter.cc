#include <gtest/gtest.h>
#include <iostream>

#include "refill/kalman_filter.h"

namespace refill {

KalmanFilter::KalmanFilter(GaussianDistribution state,
                           GaussianDistribution system_noise,
                           GaussianDistribution measurement_noise,
                           Eigen::MatrixXd sys_model, Eigen::MatrixXd obs_model)
    : state_(state),
      system_noise_(system_noise),
      measurement_noise_(measurement_noise),
      sys_model_(sys_model),
      measurement_model_(obs_model) {}

}  // namespace refill
