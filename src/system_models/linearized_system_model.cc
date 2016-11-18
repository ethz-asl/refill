#include "refill/system_models/linearized_system_model.h"

namespace refill {

LinearizedSystemModel::LinearizedSystemModel() {
}

int LinearizedSystemModel::getSystemNoiseDim() const {
  return system_noise_->mean().size();
}

}  // namespace refill
