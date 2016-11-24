#include "refill/filters/unscented_kalman_filter.h"

#include <Eigen/Dense>

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

void UnscentedKalmanFilter::predict() {}
void UnscentedKalmanFilter::update(const Eigen::VectorXd& measurement) {
  CHECK_EQ(measurement.rows(), this->measurement_model_->getMeasurementDim())
      << "Measurement has wrong dimension.";
}

// Computes unscented transform of the current system state.
void UnscentedKalmanFilter::computeUnscentedTransform() {}

}  // namespace refill
