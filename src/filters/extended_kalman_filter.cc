#include "refill/filters/extended_kalman_filter.h"

#include <Eigen/Dense>

namespace refill {

void ExtendedKalmanFilter::setState(const GaussianDistribution& state) {
  state_ = state;
}

void ExtendedKalmanFilter::predict() {
  this->predict(Eigen::VectorXd::Zero(this->system_model_->getInputDim()));
}

void ExtendedKalmanFilter::predict(const Eigen::VectorXd& input) {
  this->predict(*this->system_model_, input);
}

void ExtendedKalmanFilter::predict(const LinearizedSystemModel& system_model) {
  this->predict(system_model,
                Eigen::VectorXd::Zero(system_model.getInputDim()));
}

void ExtendedKalmanFilter::predict(const LinearizedSystemModel& system_model,
                                   const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().size());
  CHECK_EQ(system_model.getInputDim(), input.size());

  const Eigen::MatrixXd system_jacobian =
      system_model.getStateJacobian(state_.mean(), input);
  const Eigen::MatrixXd noise_jacobian =
      system_model.getNoiseJacobian(state_.mean(), input);

  CHECK_EQ(system_jacobian.rows(), system_jacobian.cols());
  CHECK_EQ(system_model.getStateDim(), system_jacobian.rows());
  CHECK_EQ(system_model.getStateDim(), noise_jacobian.rows());
  CHECK_EQ(system_model.getSystemNoiseDim(), noise_jacobian.cols());

  const Eigen::VectorXd new_state_mean =
      system_model.propagate(state_.mean(), input);
  const Eigen::MatrixXd new_state_cov =
      system_jacobian * state_.cov() * system_jacobian.transpose() +
      noise_jacobian * system_model.getSystemNoise()->cov() *
          noise_jacobian.transpose();

  state_.setDistParam(new_state_mean, new_state_cov);
}

void ExtendedKalmanFilter::update(const Eigen::VectorXd& measurement) {
  this->update(*this->measurement_model_, measurement);
}

void ExtendedKalmanFilter::update(
    const LinearizedMeasurementModel& measurement_model,
    const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement_model.getMeasurementDim(), measurement.size());
  CHECK_EQ(measurement_model.getStateDim(), state_.mean().size());

  const Eigen::MatrixXd measurement_jacobian =
      measurement_model.getMeasurementJacobian(state_.mean());
  const Eigen::MatrixXd noise_jacobian =
      measurement_model.getNoiseJacobian(state_.mean());

  CHECK_EQ(measurement_model.getMeasurementDim(), measurement_jacobian.rows());
  CHECK_EQ(measurement_model.getStateDim(), measurement_jacobian.cols());
  CHECK_EQ(measurement_model.getMeasurementDim(), noise_jacobian.rows());
  CHECK_EQ(measurement_model.getMeasurementNoiseDim(), noise_jacobian.cols());

  const Eigen::VectorXd innovation =
      measurement - measurement_model.observe(state_.mean());
  const Eigen::MatrixXd residual_cov =
      measurement_jacobian * state_.cov() * measurement_jacobian.transpose() +
      noise_jacobian * measurement_model.getMeasurementNoise()->cov() *
          noise_jacobian.transpose();
  const Eigen::MatrixXd kalman_gain =
      state_.cov() * measurement_jacobian.transpose() * residual_cov.inverse();

  const Eigen::VectorXd new_state_mean =
      state_.mean() + kalman_gain * innovation;
  const Eigen::MatrixXd new_state_cov =
      state_.cov() - kalman_gain * residual_cov * kalman_gain.transpose();

  state_.setDistParam(new_state_mean, new_state_cov);
}

}  // namespace refill
