#ifndef REFILL_FILTERS_KALMAN_FILTER_H_
#define REFILL_FILTERS_KALMAN_FILTER_H_

#include <glog/logging.h>

#include "refill/filters/filter_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic, int MEAS_DIM = Eigen::Dynamic>
class KalmanFilter : public FilterBase<STATE_DIM, MEAS_DIM> {
 public:
  KalmanFilter();
  KalmanFilter(GaussianDistribution<STATE_DIM> initial_state,
               GaussianDistribution<STATE_DIM> system_noise,
               GaussianDistribution<MEAS_DIM> measurement_noise,
               Eigen::Matrix<double, STATE_DIM, STATE_DIM> sys_model,
               Eigen::Matrix<double, MEAS_DIM, STATE_DIM> obs_model);

  void predict() {
    state_ = sys_model_ * state_ + system_noise_;
  }
  void update(Eigen::Matrix<double, MEAS_DIM, 1> measurement);

  GaussianDistribution<STATE_DIM> state() const {
    return state_;
  }

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

template<int STATE_DIM, int MEAS_DIM>
KalmanFilter<STATE_DIM, MEAS_DIM>::KalmanFilter() {
  constexpr int kCurrentSystemDim =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;

  sys_model_ =
      Eigen::Matrix<double, kCurrentSystemDim, kCurrentSystemDim>::Identity(
          kCurrentSystemDim, kCurrentSystemDim);

  // If dimensionality of measurement datastructures has dynamic size, we assume
  // it to be the same as the system model by default.
  if (MEAS_DIM == Eigen::Dynamic) {
    measurement_model_ = Eigen::Matrix<double, kCurrentSystemDim,
        kCurrentSystemDim>::Identity(kCurrentSystemDim, kCurrentSystemDim);
  } else {
    measurement_model_ =
        Eigen::Matrix<double, MEAS_DIM, kCurrentSystemDim>::Identity(
            MEAS_DIM, kCurrentSystemDim);
  }
}

template<int STATE_DIM, int MEAS_DIM>
KalmanFilter<STATE_DIM, MEAS_DIM>::KalmanFilter(
    GaussianDistribution<STATE_DIM> initial_state,
    GaussianDistribution<STATE_DIM> system_noise,
    GaussianDistribution<MEAS_DIM> measurement_noise,
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> sys_model,
    Eigen::Matrix<double, MEAS_DIM, STATE_DIM> obs_model)
    : state_(initial_state),
      system_noise_(system_noise),
      measurement_noise_(measurement_noise),
      sys_model_(sys_model),
      measurement_model_(obs_model) {
  const int state_dim = state_.dim();
  const int measurement_dim = measurement_noise.dim();

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state_dim, system_noise.dim());
    CHECK_EQ(state_dim, sys_model.cols());
    CHECK_EQ(state_dim, sys_model.rows());
    CHECK_EQ(state_dim, obs_model.cols());
  }

  if (MEAS_DIM == Eigen::Dynamic) {
    CHECK_EQ(measurement_dim, obs_model.rows());
    CHECK_EQ(measurement_dim, measurement_noise.dim());
  }
}

template<int STATE_DIM, int MEAS_DIM>
void KalmanFilter<STATE_DIM, MEAS_DIM>::update(
    Eigen::Matrix<double, MEAS_DIM, 1> measurement) {
  CHECK_EQ(measurement.size(), measurement_model_.rows());

  Eigen::Matrix<double, MEAS_DIM, 1> innovation;
  Eigen::Matrix<double, MEAS_DIM, MEAS_DIM> residual_cov;
  Eigen::Matrix<double, STATE_DIM, MEAS_DIM> kalman_gain;

  innovation = measurement - measurement_model_ * state_.mean();
  residual_cov = measurement_model_ * state_.cov()
      * measurement_model_.transpose() + measurement_noise_.cov();
  kalman_gain = state_.cov() * measurement_model_.transpose()
      * residual_cov.inverse();

  state_.setDistParam(
      state_.mean() + kalman_gain * innovation,
      state_.cov() - kalman_gain * residual_cov * kalman_gain.transpose());
}

}  // namespace refill

#endif  // REFILL_FILTERS_KALMAN_FILTER_H_
