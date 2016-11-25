#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

LinearizedMeasurementModel::LinearizedMeasurementModel(
    const std::size_t& state_dim, const std::size_t& measurement_dim,
    const DistributionInterface& measurement_noise)
    : state_dim_(state_dim),
      measurement_dim_(measurement_dim),
      measurement_noise_(measurement_noise.clone()) {
}

void LinearizedMeasurementModel::setLinearizedMeasurementModelParameters(
    const std::size_t& state_dim, const std::size_t& measurement_dim,
    const DistributionInterface& measurement_noise) {
  state_dim_ = state_dim;
  measurement_dim_ = measurement_dim;
  measurement_noise_.reset(measurement_noise.clone());
}

std::size_t LinearizedMeasurementModel::getStateDim() const {
  return state_dim_;
}

std::size_t LinearizedMeasurementModel::getMeasurementDim() const {
  return measurement_dim_;
}

std::size_t LinearizedMeasurementModel::getMeasurementNoiseDim() const {
  CHECK(measurement_noise_)
      << "[LinearizedMeasurementModel] Measurement noise has not been set.";
  return measurement_noise_->mean().size();
}

DistributionInterface* LinearizedMeasurementModel::getMeasurementNoise() const {
  CHECK(measurement_noise_)
      << "[LinearizedMeasurementModel] Measurement noise has not been set.";
  return measurement_noise_.get();
}

}  // namespace refill
