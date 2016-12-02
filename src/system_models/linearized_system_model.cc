#include "refill/system_models/linearized_system_model.h"

namespace refill {

LinearizedSystemModel::LinearizedSystemModel(
    const std::size_t& state_dim, const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {}

LinearizedSystemModel::LinearizedSystemModel(
    const std::size_t& state_dim, const DistributionInterface& system_noise,
    const std::size_t& input_dim)
    : SystemModelBase(state_dim, system_noise, input_dim) {}

}  // namespace refill
