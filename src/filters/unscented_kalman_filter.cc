#include "refill/filters/unscented_kalman_filter.h"

namespace refill {

UnscentedKalmanFilter::UnscentedKalmanFilter(const double alpha)
    : alpha_(alpha),
      state_(GaussianDistribution()),
      system_model_(nullptr),
      measurement_model_(nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const double alpha, const GaussianDistribution& initial_state)
    : alpha_(alpha),
      state_(initial_state),
      system_model_(nullptr),
      measurement_model_(nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const double alpha, const GaussianDistribution& initial_state,
    std::unique_ptr<LinearizedSystemModel> system_model,
    std::unique_ptr<LinearizedMeasurementModel> measurement_model)
    : alpha_(alpha),
      state_(initial_state),
      system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)) {
  const int kStateDimension = state_.mean().size();

  // The purpose of these checks is to verify that the dimensions of the models
  // agree with the dimension of the system state.
  CHECK_EQ(system_model_->getStateDim(), kStateDimension);
  CHECK_EQ(measurement_model_->getStateDim(), kStateDimension);
}

/** Checks whether the standard system model has been set. */
void UnscentedKalmanFilter::predict() {
  CHECK(this->system_model_) << "No default system model provided.";

  const int kInputSize = this->system_model_->getInputDim();
  this->predict(Eigen::VectorXd::Zero(kInputSize));
}

/**
 * Checks whether the standard system model has been set.
 *
 * Also checks that the input dimension match the system model input
 * dimension.
 *
 * @param input Input to the system.
 */
void UnscentedKalmanFilter::predict(const Eigen::VectorXd& input) {
  CHECK(this->system_model_) << "No default system model provided.";
  this->predict(*(this->system_model_), input);
}

/**
 * Checks whether the system model state dimension matches the filter state
 * dimension.
 *
 * @param system_model The system model to use for the prediction.
 */
void UnscentedKalmanFilter::predict(const LinearizedSystemModel& system_model) {
  const int kInputSize = system_model.getInputDim();
  this->predict(system_model, Eigen::VectorXd::Zero(kInputSize));
}

/**
 * Checks whether the system model state dimension matches the filter state
 * dimension, as well as whether the system model input dimension matches the
 * input dimension.
 *
 * @param system_model The system model to use for the prediction.
 * @param input The input to the system.
 */
void UnscentedKalmanFilter::predict(const LinearizedSystemModel& system_model,
                                    const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().size());
  CHECK_EQ(system_model.getInputDim(), input.size());

  // Generate sigma points by sampling state around mean
  Eigen::MatrixXd Sx;
  std::vector<double> S_weights;
  generateSigmaPoints(alpha_, state_, &Sx, S_weights);

  // Propagate sigma points through potentially non-linear process model
  Eigen::MatrixXd Sx_pred(Sx.rows(), Sx.cols());
  for (int i = 0; i < Sx.cols(); i++) {
    Sx_pred.col(i) = system_model.propagate(Sx.col(i), input,
                                            system_model.getNoise()->mean());
  }

  // Calc weighted mean to obtain state prediction
  Eigen::VectorXd x_pred_mean = Eigen::VectorXd::Zero(state_.mean().size());
  for (int i = 0; i < Sx_pred.cols(); i++) {
    x_pred_mean += S_weights[i] * Sx_pred.col(i);
  }

  // Predict state covariance
  Eigen::MatrixXd x_pred_cov =
      Eigen::MatrixXd::Zero(Sx_pred.rows(), x_pred_mean.size());
  for (int i = 0; i < Sx_pred.cols(); i++) {
    Eigen::VectorXd dx_i = Sx_pred.col(i) - x_pred_mean;
    x_pred_cov += S_weights[i] * dx_i * dx_i.transpose();
  }
  x_pred_cov += system_model.getNoise()->cov();

  // Update state and state cov with prediction
  state_.setDistributionParameters(x_pred_mean, x_pred_cov);
}

/**
 * Checks that the standard measurement model has been set.
 *
 * Also checks that the measurement model dimension matches the measurement
 * dimension.
 *
 * @param measurement The measurement to update the filter with.
 */
void UnscentedKalmanFilter::update(const Eigen::VectorXd& measurement) {
  CHECK(this->measurement_model_) << "No default measurement model provided.";
  this->update(*this->measurement_model_, measurement);
}

/**
 * Checks whether the measurement model dimension matches the measurement
 * dimension and whether the measurement model state dimension matches the
 * filter state dimension.
 *
 * @param measurement_model The measurement model to use for the update.
 * @param measurement The measurement to update the filter with.
 */
void UnscentedKalmanFilter::update(
    const LinearizedMeasurementModel& measurement_model,
    const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement_model.getMeasurementDim(), measurement.size());
  CHECK_EQ(measurement_model.getStateDim(), state_.mean().size());

  // Generate sigma points by sampling state around mean
  Eigen::MatrixXd Sx_pred;
  std::vector<double> S_weights;
  generateSigmaPoints(alpha_, state_, &Sx_pred, S_weights);

  // Transform sigma points to measurement space
  Eigen::MatrixXd Sy_pred(measurement.size(), Sx_pred.cols());
  for (int i = 0; i < Sx_pred.cols(); i++) {
    Sy_pred.col(i) = measurement_model.observe(
        Sx_pred.col(i), measurement_model.getNoise()->mean());
  }

  // Calc weighted mean to get measurement prediction
  Eigen::VectorXd y_pred_mean = Eigen::VectorXd::Zero(measurement.size());
  for (int i = 0; i < Sy_pred.cols(); i++) {
    y_pred_mean += S_weights[i] * Sy_pred.col(i);
  }

  // Update measurement covariance
  Eigen::MatrixXd y_pred_cov =
      Eigen::MatrixXd::Zero(Sy_pred.rows(), y_pred_mean.size());
  Eigen::MatrixXd xy_pred_cov =
      Eigen::MatrixXd::Zero(Sx_pred.rows(), y_pred_mean.size());
  for (int i = 0; i < Sy_pred.cols(); i++) {
    Eigen::VectorXd dx_i = Sx_pred.col(i) - state_.mean();
    Eigen::VectorXd dy_i = Sy_pred.col(i) - y_pred_mean;
    y_pred_cov += S_weights[i] * dy_i * dy_i.transpose();
    xy_pred_cov += S_weights[i] * dx_i * dy_i.transpose();
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
}

void UnscentedKalmanFilter::setState(const GaussianDistribution& state) {
  state_ = state;
}

/**
 * Generates a set of sigmapoints by samppling state around mean
 *
 * @param alpha: The weight of the central point (original state mean).
 * @param Sx: Ouput matrix ptr that stores the generated sigma points
 * @param S_weights: weights to be used to compute the weighted mean from sigma
 * points
 */
void UnscentedKalmanFilter::generateSigmaPoints(
    const double alpha, const GaussianDistribution& state, Eigen::MatrixXd* Sx,
    std::vector<double>& S_weights) {
  int dim = state.mean().size();
  Sx->resize(dim, 2 * dim + 1);

  // Calc matrix square root of state cov
  Eigen::MatrixXd P_sqr = state.cov().llt().matrixL();
  // Calc scaling factor to ensure sigma points preserve state scale
  double scale = sqrt(dim / (1 - alpha));

  // Set first sigma point to state mean
  Sx->col(0) = state.mean();
  // Generate two sigma points for each state dimension
  for (int i = 0; i < dim; i++) {
    Sx->col(1 + i) = state.mean() + scale * P_sqr.col(i);
    Sx->col(1 + i + dim) = state.mean() - scale * P_sqr.col(i);
  }

  S_weights.push_back(alpha);
  for (int i = 1; i < Sx->cols(); i++) {
    S_weights.push_back((1 - alpha) / (2 * dim));
  }
}

}  // namespace refill
