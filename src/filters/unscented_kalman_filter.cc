#include "refill/filters/unscented_kalman_filter.h"

#include <Eigen/Dense>

#include <cmath>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/system_models/linearized_system_model.h"

namespace refill {

UnscentedKalmanFilter::UnscentedKalmanFilter()
    : system_model_(new LinearSystemModel()),
      measurement_model_(new LinearMeasurementModel()) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const GaussianDistribution& initial_state)
    : state_(initial_state),
      system_model_(nullptr),
      measurement_model_(nullptr) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const GaussianDistribution& initial_state,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<MeasurementModelBase> measurement_model)
    : state_(initial_state),
      system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)) {
  const int state_dimension = state_.mean().size();

  CHECK_EQ(state_dimension, system_model_->getStateDim())
      << "State dimension does not agree with system model.";
  CHECK_EQ(state_dimension, measurement_model_->getStateDim())
      << "State dimension does not agree with measurement model.";
}

void UnscentedKalmanFilter::predict() {
  CHECK(system_model_) << "System model not defined.";

  const int input_size = this->system_model_->getInputDim();
  this->predict(*this->system_model_, Eigen::VectorXd::Zero(input_size));
}

void UnscentedKalmanFilter::predict(const Eigen::VectorXd& input) {
  CHECK(this->system_model_) << "No default system model provided.";
  this->predict(*this->system_model_, input);
}

void UnscentedKalmanFilter::predict(const SystemModelBase& system_model) {
  const int input_size = system_model.getInputDim();
  this->predict(system_model, Eigen::VectorXd::Zero(input_size));
}

void UnscentedKalmanFilter::predict(const SystemModelBase& system_model,
                                    const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getStateDim(), state_.mean().size());
  CHECK_EQ(system_model.getInputDim(), input.size());

  // TODO(igilitschenski): FILLME
}

void UnscentedKalmanFilter::setUkfSettings(UkfSettings settings) {
  settings_ = settings;
}

void UnscentedKalmanFilter::update(const Eigen::VectorXd& measurement) {
  this->update(*this->measurement_model_, measurement);
}

void UnscentedKalmanFilter::update(
    const MeasurementModelBase& measurement_model,
    const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement.rows(), measurement_model.getMeasurementDim())
      << "Measurement has wrong dimension.";
  CHECK_EQ(measurement_model.getStateDim(), state_.dimension())
      << "Measurement model assumes wrong state dimension.";

  // TODO(igilitschenski): FILLME
}

// This basically implements the deterministic sample set from Julier (2014),
// eq. 12. The equation of the mean sample is chosen such that all samples have
// equal weight 1/(2N+1), where N is the number of dimensions.
void UnscentedKalmanFilter::computeUnscentedTransform(
    const GaussianDistribution& noise_distribution) {
  const unsigned int noise_dimension = noise_distribution.dimension();
  unsigned int state_dimension = state_.dimension();
  Eigen::VectorXd state_mean = state_.mean();
  Eigen::MatrixXd state_covariance = state_.cov();

  if (settings_.sample_set_type == kUkfSampleJointDistribution) {
    // Augment state mean with mean from the noise.
    state_mean.conservativeResize(state_dimension + noise_dimension);
    state_mean.block(state_dimension, 0, noise_dimension, 1) =
        noise_distribution.mean();

    // Augment state covariance with noise covariance.
    state_covariance.conservativeResize(state_dimension + noise_dimension,
                                        state_dimension + noise_dimension);
    state_covariance.block(state_dimension, state_dimension, noise_dimension,
                           noise_dimension) = noise_distribution.cov();
    state_dimension += noise_dimension;
  } else {  // Sample state and noise seperately
    noise_samples_.resize(noise_dimension, 2 * noise_dimension + 1);
    noise_samples_ << Eigen::MatrixXd::Zero(noise_dimension, 1),
        Eigen::MatrixXd::Identity(noise_dimension, noise_dimension),
        -Eigen::MatrixXd::Identity(noise_dimension, noise_dimension);

    // This is the implementation of eq. (12) for the noise.
    const double noise_weights = 1 / static_cast<double>(noise_dimension);
    if (settings_.mat_square_root_algo == kUkfUseCholeskyDecomposition) {
      noise_samples_ =
          noise_distribution.mean() +
          ((noise_dimension / (1 - noise_weights)) * noise_distribution.cov())
                  .llt()
                  .matrixL() *
              state_samples_;
    } else {
      Eigen::EigenSolver<Eigen::MatrixXd> matrix_root_solver(
          noise_distribution.cov());
      const Eigen::MatrixXd eigenvalues =
          matrix_root_solver.eigenvalues().real().asDiagonal();
      const Eigen::MatrixXd covariance_root =
          matrix_root_solver.eigenvectors().real() *
          static_cast<Eigen::MatrixXd>(eigenvalues.array().sqrt());
      noise_samples_ = noise_distribution.mean() +
                       std::sqrt(noise_dimension / (1 - noise_weights)) *
                           covariance_root * state_samples_;
    }
  }
  state_samples_.resize(state_dimension, 2 * state_dimension + 1);
  state_samples_ << Eigen::MatrixXd::Zero(state_dimension, 1),
      Eigen::MatrixXd::Identity(state_dimension, state_dimension),
      -Eigen::MatrixXd::Identity(state_dimension, state_dimension);

  // This is the implementation of eq. (12) for the (joint) state.
  const double state_weights = 1 / static_cast<double>(state_dimension);
  if (settings_.mat_square_root_algo == kUkfUseCholeskyDecomposition) {
    const Eigen::MatrixXd scaled_covariance_root =
        (state_dimension / (1 - state_weights) * state_covariance)
            .llt()
            .matrixL();
    state_samples_ = state_mean.replicate(1, 2 * state_dimension + 1) +
                     scaled_covariance_root * state_samples_;
  } else {
    Eigen::EigenSolver<Eigen::MatrixXd> matrix_root_solver(state_covariance);
    const Eigen::MatrixXd eigenvalues =
        matrix_root_solver.eigenvalues().real().asDiagonal();
    Eigen::MatrixXd covariance_root =
        matrix_root_solver.eigenvectors().real() *
        static_cast<Eigen::MatrixXd>(eigenvalues.array().sqrt());
    state_samples_ = state_mean.replicate(1, 2 * state_dimension + 1) +
                     std::sqrt(state_dimension / (1 - state_weights)) *
                         covariance_root * state_samples_;
  }
}

}  // namespace refill
