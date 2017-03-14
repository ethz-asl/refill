#include "refill/system_models/linearized_system_model.h"

using std::size_t;

namespace refill {

Eigen::MatrixXd LinearizedSystemModel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(this->getStateDim(), state.rows());
  CHECK_EQ(this->getInputDim(), input.rows());

  Eigen::MatrixXd jacobian(this->getStateDim(), this->getStateDim());
  double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  Eigen::VectorXd x = state;
  Eigen::VectorXd evaluation_1(state.rows());
  Eigen::VectorXd evaluation_2(state.rows());
  for (int i = 0; i < this->getStateDim(); ++i) {
    double h = eps * std::abs(x[i]);
    if (h == 0.0) {
      h = eps;
    }
    x[i] += h;
    evaluation_1 = this->propagate(x, input, this->getSystemNoise()->mean());
    x[i] -= 2 * h;
    evaluation_2 = this->propagate(x, input, this->getSystemNoise()->mean());
    x[i] = state[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * h);
  }
  return jacobian;
}

Eigen::MatrixXd LinearizedSystemModel::getNoiseJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(this->getStateDim(), state.rows());
  CHECK_EQ(this->getInputDim(), input.rows());

  Eigen::MatrixXd jacobian(this->getSystemNoiseDim(),
                           this->getSystemNoiseDim());
  double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  constexpr Eigen::VectorXd noise_mean = this->getSystemNoise()->mean();
  Eigen::VectorXd x = noise_mean;
  Eigen::VectorXd evaluation_1(this->getSystemNoiseDim());
  Eigen::VectorXd evaluation_2(this->getSystemNoiseDim());
  for (int i = 0; i < this->getSystemNoiseDim(); ++i) {
    double h = eps * std::abs(x[i]);
    if (h == 0.0) {
      h = eps;
    }
    x[i] += h;
    evaluation_1 = this->propagate(state, input, x);
    x[i] -= 2 * h;
    evaluation_2 = this->propagate(state, input, x);
    x[i] = noise_mean[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * h);
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
