#include "refill/measurement_models/measurement_model_base.h"

using std::size_t;

namespace refill {

/**
 * Use this constructor when inheriting from this class.
 * The constructor clones the measurement noise, so it can be used again.
 *
 * @param state_dim The state dimension.
 * @param measurement_dim The measurement dimension.
 * @param measurement_noise The measurement noise.
 */
MeasurementModelBase::MeasurementModelBase(
    const size_t& state_dim, const size_t& measurement_dim,
    const DistributionInterface& measurement_noise)
    : state_dim_(state_dim),
      measurement_dim_(measurement_dim),
      measurement_noise_(measurement_noise.clone()) {}

/**
 * The function clones the measurement noise, so it can be used again.
 *
 * @param state_dim The state dimension.
 * @param measurement_dim The measurement dimension.
 * @param measurement_noise The measurement noise.
 */
void MeasurementModelBase::setMeasurementModelBaseParameters(
    const size_t& state_dim, const size_t& measurement_dim,
    const DistributionInterface& measurement_noise) {
  state_dim_ = state_dim;
  measurement_dim_ = measurement_dim;
  measurement_noise_.reset(measurement_noise.clone());
}

/** @return the state dimension. */
size_t MeasurementModelBase::getStateDim() const {
  return state_dim_;
}

/** @return the measurement dimension. */
size_t MeasurementModelBase::getMeasurementDim() const {
  return measurement_dim_;
}

/** @return the noise dimension. */
size_t MeasurementModelBase::getMeasurementNoiseDim() const {
  CHECK(measurement_noise_) << "Measurement noise has not been set.";
  return measurement_noise_->mean().size();
}

/** @return a pointer to the measurement model noise distribution. */
DistributionInterface* MeasurementModelBase::getMeasurementNoise() const {
  CHECK(measurement_noise_) << "Measurement noise has not been set.";
  return measurement_noise_.get();
}

}  // namespace refill
