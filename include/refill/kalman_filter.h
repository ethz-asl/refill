#ifndef INCLUDE_REFILL_KALMAN_FILTER_H_
#define INCLUDE_REFILL_KALMAN_FILTER_H_

#include "refill/filter_base.h"
#include "refill/gaussian_distribution.h"

namespace refill {

class KalmanFilter : public FilterBase {
 public:
  KalmanFilter();
  KalmanFilter(GaussianDistribution<> initial_state,
               GaussianDistribution<> system_noise,
               GaussianDistribution<> measurement_noise,
               Eigen::MatrixXd sys_model, Eigen::MatrixXd obs_model);

  void Predict() { state_ = sys_model_ * state_ + system_noise_; }
  void Update(Eigen::VectorXd measurement);

  GaussianDistribution<> state() const { return state_; }

 private:
  GaussianDistribution<> state_;
  GaussianDistribution<> system_noise_;
  GaussianDistribution<> measurement_noise_;
  Eigen::MatrixXd sys_model_;
  Eigen::MatrixXd measurement_model_;
};

}  // namespace refill

#endif  // INCLUDE_REFILL_KALMAN_FILTER_H_
