#include "refill/filters/extended_kalman_filter.h"

namespace refill {

/**
 * Use this constructor if you don't know the initial conditions of the state
 * or if you're not able to include them at construction.
 *
 * A use case example for this would be a filter as a member variable of a
 * class. There it often is not easy to include an initial state in the
 * constructor.
 *
 * To use the filter after using this constructor, first the state has to be
 * set using setState() and then a system / measurement model have to be
 * provided for the prediction / update step.
 */
ExtendedKalmanFilter::ExtendedKalmanFilter()
    : state_(GaussianDistribution()),
      system_model_(nullptr),
      measurement_model_(nullptr) {}

/**
 * Use this constructor if you intend to use the filter, by providing a system
 * and measurement model at every prediction / update step.
 *
 * @param initial_state The initial state of the filter.
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const GaussianDistribution& initial_state)
    : state_(initial_state),
      system_model_(nullptr),
      measurement_model_(nullptr) {}

/**
 * Use this constructor if you intend to use the filter, by using the standard
 * system and measurement models provided here.
 *
 * The ExtendedKalmanFilter class takes ownership of both models.
 *
 * Also checks whether the system and measurement model state dimensions match
 * the state dimension used in the filter.
 *
 * @param initial_state The initial state of the filter.
 * @param system_model The standard system model.
 * @param measurement_model the standard measurement model.
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const GaussianDistribution& initial_state,
    std::unique_ptr<LinearizedSystemModel> system_model,
    std::unique_ptr<LinearizedMeasurementModel> measurement_model)
    : state_(initial_state),
      system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)) {
  // The purpose of these checks is to verify that the dimensions of the models
  // agree with the dimension of the system state.
  CHECK_EQ(system_model_->getStateDim(), state_.mean().rows());
  CHECK_EQ(measurement_model_->getStateDim(), state_.mean().rows());
}

/** @param state The new filter state. */
void ExtendedKalmanFilter::setState(const GaussianDistribution& state) {
  state_ = state;
}

/** Checks whether the standard system model has been set. */
void ExtendedKalmanFilter::predict() {
  CHECK(this->system_model_) << "No default system model provided.";

  this->predict(Eigen::VectorXd::Zero(this->system_model_->getInputDim()));
}

/**
 * Checks whether the standard system model has been set.
 *
 * Also checks that the input dimension match the system model input
 * dimension.
 *
 * @param input Input to the system.
 */
void ExtendedKalmanFilter::predict(const Eigen::VectorXd& input) {
  CHECK(this->system_model_) << "No default system model provided.";
  this->predict(*(this->system_model_), input);
}

/**
 * Checks whether the system model state dimension matches the filter state
 * dimension.
 *
 * @param system_model The system model to use for the prediction.
 */
void ExtendedKalmanFilter::predict(const LinearizedSystemModel& system_model) {
  this->predict(system_model,
                Eigen::VectorXd::Zero(system_model.getInputDim()));
}

/**
 * Checks whether the system model state dimension matches the filter state
 * dimension, as well as whether the system model input dimension matches the
 * input dimension.
 *
 * @param system_model The system model to use for the prediction.
 * @param input The input to the system.
 */
void ExtendedKalmanFilter::predict(const LinearizedSystemModel& system_model,
                                   const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().rows());
  CHECK_EQ(system_model.getInputDim(), input.rows());

  const Eigen::MatrixXd system_jacobian =
      system_model.getStateJacobian(state_.mean(), input);
  const Eigen::MatrixXd noise_jacobian =
      system_model.getNoiseJacobian(state_.mean(), input);

  const Eigen::VectorXd new_state_mean = system_model.propagate(
      state_.mean(), input, system_model.getNoise()->mean());

  // Use .selfadjointView<>() to guarantee symmetric matrix
  const Eigen::MatrixXd new_state_cov =
      (system_jacobian * state_.cov() * system_jacobian.transpose() +
      noise_jacobian * system_model.getNoise()->cov() *
      noise_jacobian.transpose()).selfadjointView<Eigen::Upper>();

  state_.setDistributionParameters(new_state_mean, new_state_cov);
}

/**
 * Checks that the standard measurement model has been set.
 *
 * Also checks that the measurement model dimension matches the measurement
 * dimension.
 *
 * @param measurement The measurement to update the filter with.
 */
void ExtendedKalmanFilter::update(const Eigen::VectorXd& measurement) {
  CHECK(this->measurement_model_) << "No default measurement model provided.";
  this->update(*(this->measurement_model_), measurement);
}

/**
 * Checks whether the measurement model dimension matches the measurement
 * dimension and whether the measurement model state dimension matches the
 * filter state dimension.
 *
 * Also checks that the residual covariance is invertible.
 *
 * @param measurement_model The measurement model to use for the update.
 * @param measurement The measurement to update the filter with.
 */
void ExtendedKalmanFilter::update(
    const LinearizedMeasurementModel& measurement_model,
    const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement_model.getMeasurementDim(), measurement.rows());
  CHECK_EQ(measurement_model.getStateDim(), state_.mean().rows());

  const Eigen::MatrixXd measurement_jacobian =
      measurement_model.getMeasurementJacobian(state_.mean());
  const Eigen::MatrixXd noise_jacobian =
      measurement_model.getNoiseJacobian(state_.mean());

  const Eigen::MatrixXd measurement_noise_cov = noise_jacobian
      * measurement_model.getNoise()->cov() * noise_jacobian.transpose();

  const Eigen::VectorXd innovation = measurement
      - measurement_model.observe(state_.mean(),
                                  measurement_model.getNoise()->mean());
  const Eigen::MatrixXd residual_cov =
      measurement_jacobian * state_.cov() * measurement_jacobian.transpose() +
      measurement_noise_cov;

  // Use of LU decomposition with complete pivoting for computing the inverse
  // of the residual covariance within Kalman gain computation enables us to
  // perform invertability checks.
  Eigen::FullPivLU<Eigen::MatrixXd> residual_cov_lu(residual_cov);
  CHECK(residual_cov_lu.isInvertible())
      << "Residual covariance is not invertible.";
  const Eigen::MatrixXd kalman_gain = state_.cov() *
                                      measurement_jacobian.transpose() *
                                      residual_cov.inverse();

  const Eigen::VectorXd new_state_mean =
      state_.mean() + kalman_gain * innovation;

  const Eigen::MatrixXd cov_scaling = Eigen::MatrixXd::Identity(
      state_.mean().rows(), state_.mean().rows())
      - kalman_gain * measurement_jacobian;

  // Use .selfadjointView<>() to guarantee symmetric matrix
  // Use Joseph form for better numerical stability
  const Eigen::MatrixXd new_state_cov =
      (cov_scaling * state_.cov() * cov_scaling.transpose()
      + kalman_gain * measurement_noise_cov * kalman_gain.transpose())
      .selfadjointView<Eigen::Upper>();

  state_.setDistributionParameters(new_state_mean, new_state_cov);
}

}  // namespace refill
