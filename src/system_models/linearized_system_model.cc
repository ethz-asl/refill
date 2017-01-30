#include "refill/system_models/linearized_system_model.h"

using std::size_t;

namespace refill {

/**
 * Use this constructor if your system model does not have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 */
LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {}

/**
 * Use this constructor if your system does have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 * @param input_dim The systems input dimension.
 */
LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise,
    const size_t& input_dim)
    : SystemModelBase(state_dim, system_noise, input_dim) {}

}  // namespace refill
