#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

LinearizedMeasurementModel::LinearizedMeasurementModel(
    const unsigned int& state_dim, const unsigned int& measurement_dim,
    const DistributionInterface& measurement_noise)
    : state_dim_(state_dim),
      measurement_dim_(measurement_dim),
      measurement_noise_(measurement_noise.clone()) {
}

void LinearizedMeasurementModel::setLinearizedMeasurementModelParameters(
    const unsigned int& state_dim, const unsigned int& measurement_dim,
    const DistributionInterface& measurement_noise) {
  state_dim_ = state_dim;
  measurement_dim_ = measurement_dim;
  measurement_noise_.reset(measurement_noise.clone());
}

unsigned int LinearizedMeasurementModel::getStateDim() const {
  return state_dim_;
}

unsigned int LinearizedMeasurementModel::getMeasurementDim() const {
  return measurement_dim_;
}

unsigned int LinearizedMeasurementModel::getMeasurementNoiseDim() const {
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
