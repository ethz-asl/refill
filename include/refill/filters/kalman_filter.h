#ifndef REFILL_FILTERS_KALMAN_FILTER_H_
#define REFILL_FILTERS_KALMAN_FILTER_H_

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"

namespace refill {

template <int STATE_DIM = Eigen::Dynamic, int MEAS_DIM = Eigen::Dynamic>
class KalmanFilter : public FilterBase<STATE_DIM, MEAS_DIM> {
 public:
  KalmanFilter();
  KalmanFilter(GaussianDistribution<STATE_DIM> initial_state,
               GaussianDistribution<STATE_DIM> system_noise,
               GaussianDistribution<MEAS_DIM> measurement_noise,
               Eigen::Matrix<double, STATE_DIM, STATE_DIM> sys_model,
               Eigen::Matrix<double, MEAS_DIM, STATE_DIM> obs_model);

  void Predict() { state_ = sys_model_ * state_ + system_noise_; }
  void Update(Eigen::Matrix<double, MEAS_DIM, 1> measurement);

  GaussianDistribution<STATE_DIM> state() const { return state_; }

 private:
  GaussianDistribution<STATE_DIM> state_;
  GaussianDistribution<STATE_DIM> system_noise_;
  GaussianDistribution<MEAS_DIM> measurement_noise_;
  Eigen::Matrix<double, STATE_DIM, STATE_DIM> sys_model_;
  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> measurement_model_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using KalmanFilterXd = KalmanFilter<Eigen::Dynamic, Eigen::Dynamic>;

}  // namespace refill

#include "./kalman_filter-inl.h"

#endif  // REFILL_FILTERS_KALMAN_FILTER_H_
