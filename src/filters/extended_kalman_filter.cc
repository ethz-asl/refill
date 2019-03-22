#include "refill/filters/extended_kalman_filter.h"
#include <iostream>

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
      FilterBase(nullptr, nullptr) {}

/**
 * Use this constructor if you intend to use the filter with a default system
 * model, by providing a measurement model at every prediction / update step.
 * You do not know the inital state.
 *
 * @param system_model The standard system model
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
    std::unique_ptr<LinearizedSystemModel> system_model)
    : state_(GaussianDistribution()),
      FilterBase(std::move(system_model), nullptr) {}

/**
 * Use this constructor if you intend to use the filter, by providing a system
 * and measurement model at every prediction / update step.
 *
 * @param initial_state The initial state of the filter.
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const GaussianDistribution& initial_state)
    : state_(initial_state),
      FilterBase(nullptr, nullptr) {}

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
      FilterBase(std::move(system_model), std::move(measurement_model)) {
  const int kStateDimension = state_.mean().size();

  // The purpose of these checks is to verify that the dimensions of the models
  // agree with the dimension of the system state.
  CHECK_EQ(system_model_->getStateDim(), kStateDimension);
  CHECK_EQ(measurement_model_->getStateDim(), kStateDimension);
}

/** @param state The new filter state. */
void ExtendedKalmanFilter::setState(const GaussianDistribution& state) {
  state_ = state;
}

/**
 * Checks whether the system model state dimension matches the filter state
 * dimension, as we
 ll as whether the system model input dimension matches the
 * input dimension.
 *
 * @param system_model The system model to use for the prediction.
 * @param input The input to the system.
 */
void ExtendedKalmanFilter::predict(const double dt,
                                   SystemModelBase& system_model,
                                   const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().size());
  CHECK_EQ(system_model.getInputDim(), input.size());

  system_model.setDeltaT(dt);

  const Eigen::MatrixXd system_jacobian =
      system_model.getStateJacobian(state_.mean(), input);
  const Eigen::MatrixXd noise_jacobian =
      system_model.getNoiseJacobian(state_.mean(), input);

  // Defining temporary matrices for transpose, since Eigen v3.2.10 exhibits a
  // bug where matrix multiplications with a transpose halts the program
  // execution here. Reason is unknown.
  const Eigen::MatrixXd system_jacobian_transpose = system_jacobian.transpose();
  const Eigen::MatrixXd noise_jacobian_transpose = noise_jacobian.transpose();

  const Eigen::VectorXd new_state_mean = system_model.propagate(
      state_.mean(), input, system_model.getNoise()->mean());

  // Use .selfadjointView<>() to guarantee symmetric matrix
  const Eigen::MatrixXd new_state_cov =
      (system_jacobian * state_.cov() * system_jacobian_transpose +
       noise_jacobian * system_model.getNoise()->cov() *
           noise_jacobian_transpose)
          .selfadjointView<Eigen::Upper>();

  state_.setDistributionParameters(new_state_mean, new_state_cov);
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
    const MeasurementModelBase& measurement_model,
    const Eigen::VectorXd& measurement, double* likelihood) {
  CHECK_EQ(measurement_model.getMeasurementDim(), measurement.size());
  CHECK_EQ(measurement_model.getStateDim(), state_.mean().size());

  const Eigen::MatrixXd measurement_jacobian =
      measurement_model.getMeasurementJacobian(state_.mean());
  const Eigen::MatrixXd noise_jacobian =
      measurement_model.getNoiseJacobian(state_.mean());

  // Defining temporary matrices for transpose, since Eigen v3.2.10 exhibits a
  // bug where matrix multiplications with a transpose halts the program
  // execution here. Reason is unknown.
  const Eigen::MatrixXd measurement_jacobian_transpose =
      measurement_jacobian.transpose();
  const Eigen::MatrixXd noise_jacobian_transpose = noise_jacobian.transpose();

  const Eigen::MatrixXd measurement_noise_cov =
      noise_jacobian * measurement_model.getNoise()->cov() *
      noise_jacobian_transpose;

  const Eigen::VectorXd innovation =
      measurement -
      measurement_model.observe(state_.mean(),
                                measurement_model.getNoise()->mean());

  const Eigen::MatrixXd residual_cov =
      measurement_jacobian * state_.cov() * measurement_jacobian_transpose +
      measurement_noise_cov;

  // Use of LU decomposition with complete pivoting for computing the inverse
  // of the residual covariance within Kalman gain computation enables us to
  // perform invertability checks.
  Eigen::FullPivLU<Eigen::MatrixXd> residual_cov_lu(residual_cov);
  CHECK(residual_cov_lu.isInvertible())
      << "Residual covariance is not invertible.";
  const Eigen::MatrixXd kalman_gain =
      state_.cov() * measurement_jacobian_transpose * residual_cov.inverse();

  // Defining temporary matrix for transpose, since Eigen v3.2.10 exhibits a
  // bug where matrix multiplications with a transpose halts the program
  // execution here. Reason is unknown.
  const Eigen::MatrixXd kalman_gain_transpose = kalman_gain.transpose();

  const Eigen::VectorXd new_state_mean =
      state_.mean() + kalman_gain * innovation;

  const Eigen::MatrixXd cov_scaling =
      Eigen::MatrixXd::Identity(state_.mean().rows(), state_.mean().rows()) -
      kalman_gain * measurement_jacobian;

  // Use .selfadjointView<>() to guarantee symmetric matrix
  // Use Joseph form for better numerical stability
  const Eigen::MatrixXd new_state_cov =
      (cov_scaling * state_.cov() * cov_scaling.transpose() +
       kalman_gain * measurement_noise_cov * kalman_gain.transpose())
          .selfadjointView<Eigen::Upper>();

  state_.setDistributionParameters(new_state_mean, new_state_cov);

  refill::GaussianDistribution probabablity_density(
      Eigen::VectorXd::Zero(measurement_model.getMeasurementDim()),
      residual_cov);
  *likelihood = probabablity_density.evaluatePdf(innovation);
}

}  // namespace refill
