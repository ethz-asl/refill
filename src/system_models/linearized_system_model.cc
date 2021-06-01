#include "refill/system_models/linearized_system_model.h"

using std::size_t;

namespace refill {


/**
 * Uses a second order central finite difference approximation for the
 * Jacobian computation, if not overloaded.
 *
 * @param state The current system state.
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system state @f$x_k@f$.
 */
Eigen::MatrixXd LinearizedSystemModel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(state.rows(), this->getStateDim());
  CHECK_EQ(input.rows(), this->getInputDim());

  Eigen::MatrixXd jacobian(this->getStateDim(), this->getStateDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  Eigen::VectorXd diff_coeff = state;
  Eigen::VectorXd evaluation_1(this->getStateDim());
  Eigen::VectorXd evaluation_2(this->getStateDim());
  for (std::size_t i = 0; i < this->getStateDim(); ++i) {
    double step_size = eps * std::abs(diff_coeff[i]);
    if (step_size == 0.0) {
      step_size = eps;
    }
    diff_coeff[i] += step_size;
    evaluation_1 = this->propagate(diff_coeff, input,
                                   this->getNoise()->mean());
    diff_coeff[i] -= 2 * step_size;
    evaluation_2 = this->propagate(diff_coeff, input,
                                   this->getNoise()->mean());
    diff_coeff[i] = state[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * step_size);
  }
  return jacobian;
}

/**
 * Uses a second order central finite difference approximation for the
 * Jacobian computation, if not overloaded.
 *
 * @param state The current system state.
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system noise @f$v_k@f$.
 */
Eigen::MatrixXd LinearizedSystemModel::getNoiseJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(state.rows(), this->getStateDim());
  CHECK_EQ(input.rows(), this->getInputDim());

  Eigen::MatrixXd jacobian(this->getStateDim(), this->getNoiseDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  const Eigen::VectorXd noise_mean = this->getNoise()->mean();
  Eigen::VectorXd diff_coeff = noise_mean;
  Eigen::VectorXd evaluation_1(this->getStateDim());
  Eigen::VectorXd evaluation_2(this->getStateDim());
  for (std::size_t i = 0; i < this->getNoiseDim(); ++i) {
    double step_size = eps * std::abs(diff_coeff[i]);
    if (step_size == 0.0) {
      step_size = eps;
    }
    diff_coeff[i] += step_size;
    evaluation_1 = this->propagate(state, input, diff_coeff);
    diff_coeff[i] -= 2 * step_size;
    evaluation_2 = this->propagate(state, input, diff_coeff);
    diff_coeff[i] = noise_mean[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * step_size);
  }
  return jacobian;
}

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
