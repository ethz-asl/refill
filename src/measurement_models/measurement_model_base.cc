#include "refill/measurement_models/measurement_model_base.h"

using std::size_t;

namespace refill {

MeasurementModelBase::MeasurementModelBase(
    const size_t& state_dim, const size_t& measurement_dim,
    const DistributionInterface& measurement_noise)
    : state_dim_(state_dim),
      measurement_dim_(measurement_dim),
      measurement_noise_(measurement_noise.clone()) {
}

void MeasurementModelBase::setMeasurementModelBaseParameters(
    const size_t& state_dim, const size_t& measurement_dim,
    const DistributionInterface& measurement_noise) {
  state_dim_ = state_dim;
  measurement_dim_ = measurement_dim;
  measurement_noise_.reset(measurement_noise.clone());
}

size_t MeasurementModelBase::getStateDim() const {
  return state_dim_;
}

size_t MeasurementModelBase::getMeasurementDim() const {
  return measurement_dim_;
}

size_t MeasurementModelBase::getMeasurementNoiseDim() const {
  CHECK(measurement_noise_) << "Measurement noise has not been set.";
  return measurement_noise_->mean().size();
}

DistributionInterface* MeasurementModelBase::getMeasurementNoise() const {
  CHECK(measurement_noise_) << "Measurement noise has not been set.";
  return measurement_noise_.get();
}

}  // namespace refill
