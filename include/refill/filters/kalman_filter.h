#ifndef REFILL_FILTERS_KALMAN_FILTER_H_
#define REFILL_FILTERS_KALMAN_FILTER_H_

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"

namespace refill {

template <int STATEDIM = Eigen::Dynamic, int MEASDIM = Eigen::Dynamic>
class KalmanFilter : public FilterBase<STATEDIM, MEASDIM> {
 public:
  KalmanFilter();
  KalmanFilter(GaussianDistribution<STATEDIM> initial_state,
               GaussianDistribution<STATEDIM> system_noise,
               GaussianDistribution<MEASDIM> measurement_noise,
               Eigen::Matrix<double, STATEDIM, STATEDIM> sys_model,
               Eigen::Matrix<double, MEASDIM, STATEDIM> obs_model);

  void Predict() { state_ = sys_model_ * state_ + system_noise_; }
  void Update(Eigen::Matrix<double, MEASDIM, 1> measurement);

  GaussianDistribution<STATEDIM> state() const { return state_; }

 private:
  GaussianDistribution<STATEDIM> state_;
  GaussianDistribution<STATEDIM> system_noise_;
  GaussianDistribution<MEASDIM> measurement_noise_;
  Eigen::Matrix<double, STATEDIM, STATEDIM> sys_model_;
  Eigen::Matrix<double, MEASDIM, STATEDIM> measurement_model_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using KalmanFilterXd = KalmanFilter<Eigen::Dynamic, Eigen::Dynamic>;

}  // namespace refill

#include "./kalman_filter-inl.h"

#endif  // REFILL_FILTERS_KALMAN_FILTER_H_
