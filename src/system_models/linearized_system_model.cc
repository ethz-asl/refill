#include "refill/system_models/linearized_system_model.h"

using std::size_t;

namespace refill {

LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {}

LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise,
    const size_t& input_dim)
    : SystemModelBase(state_dim, system_noise, input_dim) {}

}  // namespace refill
