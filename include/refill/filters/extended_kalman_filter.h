#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/filters/filter_base.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic>
class ExtendedKalmanFilter : public FilterBase<STATE_DIM,
    ExtendedKalmanFilter<STATE_DIM>> {
 public:
  ExtendedKalmanFilter();
  explicit ExtendedKalmanFilter(
      const GaussianDistribution<STATE_DIM>& initial_state);

  template<int INPUT_DIM = Eigen::Dynamic>
  void predict(
      const SystemModelBase<STATE_DIM, INPUT_DIM>& system_model,
      const Eigen::Matrix<double, INPUT_DIM, 1>& input = Eigen::Matrix<double,
          INPUT_DIM, 1>::Zero(INPUT_DIM, 1));

  template<int MEAS_DIM = Eigen::Dynamic>
  void update(
      const MeasurementModelBase<STATE_DIM, MEAS_DIM>& measurement_model,
      const Eigen::Matrix<double, MEAS_DIM, 1>& measurement);

  GaussianDistribution<STATE_DIM> state() const {
    return state_;
  }

 private:
  GaussianDistribution<STATE_DIM> state_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using ExtendedKalmanFilterXd = ExtendedKalmanFilter<>;

// Function implementations

template<int STATE_DIM>
ExtendedKalmanFilter<STATE_DIM>::ExtendedKalmanFilter() {
}

template<int STATE_DIM>
ExtendedKalmanFilter<STATE_DIM>::ExtendedKalmanFilter(
    const GaussianDistribution<STATE_DIM>& initial_state)
    : state_(initial_state) {
}

template<int STATE_DIM>
template<int INPUT_DIM>
void ExtendedKalmanFilter<STATE_DIM>::predict(
    const SystemModelBase<STATE_DIM, INPUT_DIM>& system_model,
    const Eigen::Matrix<double, INPUT_DIM, 1>& input) {

  Eigen::Matrix<double, STATE_DIM, STATE_DIM> system_mat;
  system_mat = system_model.getJacobian();

  state_.setDistParam(
      system_model.propagate(state_.mean(), input),
      system_mat * state_.cov() * system_mat.transpose()
          + system_model.getSystemNoise()->cov());
}

template<int STATE_DIM>
template<int MEAS_DIM>
void ExtendedKalmanFilter<STATE_DIM>::update(
    const MeasurementModelBase<STATE_DIM, MEAS_DIM>& measurement_model,
    const Eigen::Matrix<double, MEAS_DIM, 1>& measurement) {
  CHECK_EQ(measurement.size(), measurement_model.getMeasurementDim());

  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> measurement_mat;
  measurement_mat = measurement_model.getJacobian();

  if (MEAS_DIM == Eigen::Dynamic) {
    CHECK_EQ(measurement.size(), measurement_model.getMeasurementDim());
  }

  Eigen::Matrix<double, MEAS_DIM, 1> innovation;
  Eigen::Matrix<double, MEAS_DIM, MEAS_DIM> residual_cov;
  Eigen::Matrix<double, STATE_DIM, MEAS_DIM> kalman_gain;

  innovation = measurement - measurement_model.observe(state_.mean());
  residual_cov = measurement_mat * state_.cov() * measurement_mat.transpose()
      + measurement_model.getMeasurementNoise()->cov();
  kalman_gain = state_.cov() * measurement_mat.transpose()
      * residual_cov.inverse();

  state_.setDistParam(
      state_.mean() + kalman_gain * innovation,
      state_.cov() - kalman_gain * residual_cov * kalman_gain.transpose());
}

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
