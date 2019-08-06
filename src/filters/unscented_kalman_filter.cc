#include "refill/filters/unscented_kalman_filter.h"
#include <iostream>

namespace refill {

UnscentedKalmanFilter::UnscentedKalmanFilter()
    : state_(GaussianDistribution()), FilterBase(nullptr, nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const GaussianDistribution& initial_state)
    : state_(initial_state), FilterBase(nullptr, nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    std::unique_ptr<SystemModelBase> system_model)
    : state_(GaussianDistribution()),
      FilterBase(std::move(system_model), nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const GaussianDistribution& initial_state,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<MeasurementModelBase> measurement_model)
    : state_(initial_state),
      FilterBase(std::move(system_model), std::move(measurement_model)) {
  const int kStateDimension = state_.mean().size();

  // The purpose of these checks is to verify that the dimensions of the models
  // agree with the dimension of the system state.
  CHECK_EQ(system_model_->getStateDim(), kStateDimension);
  CHECK_EQ(measurement_model_->getStateDim(), kStateDimension);
}

void UnscentedKalmanFilter::predict(const double stamp,
                                    SystemModelBase& system_model,
                                    const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().size());
  CHECK_EQ(system_model.getInputDim(), input.size());

  double dt = stamp - state_stamp_;
  CHECK(dt >= 0) << "Negative dt: Cannot perform prediction!";
  system_model.setTimeStamp(dt, stamp);

  // Generate sigma points by sampling state around mean
  Eigen::MatrixXd sigma_states;
  std::vector<double> sigma_weights;
  generateSigmaPoints(&sigma_states, &sigma_weights, kUkfAlpha, state_);

  // Propagate sigma points through potentially non-linear process model
  Eigen::MatrixXd simga_points_pred(sigma_states.rows(), sigma_states.cols());
  for (int i = 0; i < sigma_states.cols(); i++) {
    simga_points_pred.col(i) = system_model.propagate(
        sigma_states.col(i), input, system_model.getNoise()->mean());
  }

  // Calc weighted mean to obtain state prediction
  Eigen::VectorXd x_pred_mean = Eigen::VectorXd::Zero(state_.mean().size());
  for (int i = 0; i < simga_points_pred.cols(); i++) {
    x_pred_mean += sigma_weights[i] * simga_points_pred.col(i);
  }

  // Predict state covariance
  Eigen::MatrixXd x_pred_cov =
      Eigen::MatrixXd::Zero(simga_points_pred.rows(), x_pred_mean.size());
  for (int i = 0; i < simga_points_pred.cols(); i++) {
    Eigen::VectorXd dx_i = simga_points_pred.col(i) - x_pred_mean;
    x_pred_cov += sigma_weights[i] * dx_i * dx_i.transpose();
  }
  x_pred_cov += system_model.getNoise()->cov();

  // Update state and state cov with prediction
  state_.setDistributionParameters(x_pred_mean, x_pred_cov);
  state_stamp_ = stamp;
}

void UnscentedKalmanFilter::update(
    const MeasurementModelBase& measurement_model,
    const Eigen::VectorXd& measurement, double* likelihood) {
  CHECK_EQ(measurement_model.getMeasurementDim(), measurement.size());
  CHECK_EQ(measurement_model.getStateDim(), state_.mean().size());

  // Generate sigma points by sampling state around mean
  Eigen::MatrixXd sigma_states_pred;
  std::vector<double> sigma_weights;
  generateSigmaPoints(&sigma_states_pred, &sigma_weights, kUkfAlpha, state_);
  // Transform sigma points to measurement space
  Eigen::MatrixXd sigma_measurements_pred(measurement.size(),
                                          sigma_states_pred.cols());
  for (int i = 0; i < sigma_states_pred.cols(); i++) {
    sigma_measurements_pred.col(i) = measurement_model.observe(
        sigma_states_pred.col(i), measurement_model.getNoise()->mean());
  }

  // Calc weighted mean to get measurement prediction
  Eigen::VectorXd y_pred_mean = Eigen::VectorXd::Zero(measurement.size());
  for (int i = 0; i < sigma_measurements_pred.cols(); i++) {
    y_pred_mean += sigma_weights[i] * sigma_measurements_pred.col(i);
  }

  // Update measurement covariance
  Eigen::MatrixXd y_pred_cov =
      Eigen::MatrixXd::Zero(sigma_measurements_pred.rows(), y_pred_mean.size());
  Eigen::MatrixXd xy_pred_cov =
      Eigen::MatrixXd::Zero(sigma_states_pred.rows(), y_pred_mean.size());
  for (int i = 0; i < sigma_measurements_pred.cols(); i++) {
    Eigen::VectorXd dx_i = sigma_states_pred.col(i) - state_.mean();
    Eigen::VectorXd dy_i = sigma_measurements_pred.col(i) - y_pred_mean;
    y_pred_cov += sigma_weights[i] * dy_i * dy_i.transpose();
    xy_pred_cov += sigma_weights[i] * dx_i * dy_i.transpose();
  }
  y_pred_cov += measurement_model.getNoise()->cov();

  // Check invertability of measurement cov
  Eigen::FullPivLU<Eigen::MatrixXd> y_pred_cov_lu(y_pred_cov);
  CHECK(y_pred_cov_lu.isInvertible())
      << "Residual covariance is not invertible.";

  // Calcuate kalman gain and innovation
  const Eigen::MatrixXd kalman_gain = xy_pred_cov * y_pred_cov.inverse();
  const Eigen::VectorXd innovation = measurement - y_pred_mean;

  // Update state and state covariance
  const Eigen::VectorXd updated_state_mean =
      state_.mean() + kalman_gain * innovation;
  const Eigen::MatrixXd updated_state_cov =
      state_.cov() - kalman_gain * y_pred_cov * kalman_gain.transpose();

  state_.setDistributionParameters(updated_state_mean, updated_state_cov);

  refill::GaussianDistribution residual_pdf(
      Eigen::VectorXd::Zero(measurement_model.getMeasurementDim()), y_pred_cov);
  *likelihood = residual_pdf.evaluatePdf(innovation);
}

/**
 * Generates a set of sigmapoints by sampling state around mean
 *
 * @param alpha: The weight of the central point (original state mean).
 * @param Sx: Ouput matrix ptr that stores the generated sigma points
 * @param S_weights: weights to be used to compute the weighted mean from sigma
 * points
 */
void UnscentedKalmanFilter::generateSigmaPoints(
    Eigen::MatrixXd* sigma_states, std::vector<double>* simga_weights,
    const double alpha, const GaussianDistribution& state) {
  int dim = state.mean().size();
  sigma_states->resize(dim, 2 * dim + 1);

  // Calc matrix square root of state cov
  Eigen::MatrixXd P_sqr = state.cov().llt().matrixL();
  // Calc scaling factor to ensure sigma points preserve state scale
  double scale = sqrt(dim / (1 - alpha));

  // Set first sigma point to state mean
  sigma_states->col(0) = state.mean();
  // Generate two sigma points for each state dimension
  for (int i = 0; i < dim; i++) {
    sigma_states->col(1 + i) = state.mean() + scale * P_sqr.col(i);
    sigma_states->col(1 + i + dim) = state.mean() - scale * P_sqr.col(i);
  }

  simga_weights->push_back(alpha);
  for (int i = 1; i < sigma_states->cols(); i++) {
    simga_weights->push_back((1 - alpha) / (2 * dim));
  }
}

void UnscentedKalmanFilter::setState(const GaussianDistribution& state) {
  state_ = state;
}
}  // namespace refill
