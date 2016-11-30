#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

/**
 * Use this constructor to create a linearized measurement model.
 * The constructor clones the measurement noise, so it can be used again.
 *
 * @param state_dim The measurement models state dimension.
 * @param measurement_dim The measurement models measurement dimension.
 * @param measurement_noise The measurement noise.
 */
LinearizedMeasurementModel::LinearizedMeasurementModel(
    const std::size_t& state_dim, const std::size_t& measurement_dim,
    const DistributionInterface& measurement_noise)
    : MeasurementModelBase(state_dim, measurement_dim, measurement_noise) {}

}  // namespace refill
