#include "refill/filters/extended_kalman_filter.h"

namespace refill {

// If standard constructor is called, assume one dimensional system.
ExtendedKalmanFilter::ExtendedKalmanFilter()
    : state_(1) {
}

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const GaussianDistribution& initial_state)
    : state_(initial_state) {
}

void ExtendedKalmanFilter::predict(
    const LinearizedSystemModel& system_model) {
  this->predict(system_model,
                Eigen::VectorXd::Zero(system_model.getInputDim()));
}

void ExtendedKalmanFilter::predict(const LinearizedSystemModel& system_model,
                                   const Eigen::VectorXd& input) {
  const Eigen::MatrixXd system_mat = system_model.getJacobian();

  // TODO(jwidauer): Implement noise matrix.
  Eigen::VectorXd new_state_mean = system_model.propagate(state_.mean(), input);
  Eigen::MatrixXd new_state_cov = system_mat * state_.cov()
      * system_mat.transpose() + system_model.getSystemNoise()->cov();

  state_.setDistParam(new_state_mean, new_state_cov);
}

void ExtendedKalmanFilter::update(
    const LinearizedMeasurementModel& measurement_model,
    const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement.size(), measurement_model.getMeasurementDim());

  const Eigen::MatrixXd measurement_mat = measurement_model.getJacobian();

  const Eigen::VectorXd innovation = measurement
      - measurement_model.observe(state_.mean());
  const Eigen::MatrixXd residual_cov = measurement_mat * state_.cov()
      * measurement_mat.transpose()
      + measurement_model.getMeasurementNoise()->cov();
  const Eigen::MatrixXd kalman_gain = state_.cov() * measurement_mat.transpose()
      * residual_cov.inverse();

  Eigen::VectorXd new_state_mean = state_.mean() + kalman_gain * innovation;
  Eigen::MatrixXd new_state_cov = state_.cov()
      - kalman_gain * residual_cov * kalman_gain.transpose();

  state_.setDistParam(new_state_mean, new_state_cov);
}

}  // namespace refill
