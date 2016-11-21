#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

int LinearizedMeasurementModel::getMeasurementNoiseDim() const {
  return measurement_noise_->mean().size();
}

}
// namespace refill
