#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

#include "gaussian_distribution.h"

namespace refill {

class KalmanFilter {
 public:
  KalmanFilter(GaussianDistribution initial_state,
               GaussianDistribution system_noise,
               GaussianDistribution measurement_noise,
               Eigen::MatrixXd sys_model, Eigen::MatrixXd obs_model);

 private:
  GaussianDistribution state_;
  GaussianDistribution system_noise_;
  GaussianDistribution measurement_noise_;
  Eigen::MatrixXd sys_model_;
  Eigen::MatrixXd measurement_model_;
};

}  // namespace refill

#endif  // KALMANFILTER_H_
