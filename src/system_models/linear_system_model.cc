#include "refill/system_models/linear_system_model.h"

namespace refill {

// If standard constructor is called, we assume a one dimensional system
// without input.
LinearSystemModel::LinearSystemModel()
    : system_mapping_(Eigen::MatrixXd::Identity(1, 1)),
      input_mapping_(Eigen::MatrixXd::Zero(0, 0)),
      system_noise_(new GaussianDistribution()) {
}

// If constructor is called without input matrix, we assume there is no input
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mat,
                                     const DistributionInterface& system_noise)
    : system_mapping_(system_mat),
      system_noise_(system_noise.clone()),
      input_mapping_(Eigen::MatrixXd::Zero(0, 0)) {
  CHECK_EQ(system_mat.rows(), system_mat.cols());
  CHECK_EQ(system_mat.rows(), system_noise.mean().size());
}

LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mat,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mat)
    : system_mapping_(system_mat),
      input_mapping_(input_mat),
      system_noise_(system_noise.clone()) {
  CHECK_EQ(system_mat.rows(), system_mat.cols());
  CHECK_EQ(system_mat.rows(), input_mat.rows());
  CHECK_EQ(system_mat.rows(), system_noise.mean().size());
}

void LinearSystemModel::setSystemParameters(
    const Eigen::MatrixXd& system_mat,
    const DistributionInterface& system_noise) {
  this->setSystemParameters(system_mat, system_noise,
                            Eigen::MatrixXd::Zero(0, 0));
}

void LinearSystemModel::setSystemParameters(
    const Eigen::MatrixXd& system_mat,
    const DistributionInterface& system_noise,
    const Eigen::MatrixXd& input_mat) {
  CHECK_EQ(system_mat.rows(), system_mat.cols());
  CHECK_EQ(system_mat.rows(), system_noise.mean().size());

  if (input_mat.size() != 0) {
    CHECK_EQ(system_mat.rows(), input_mat.rows());
  }

  system_mapping_ = system_mat;
  system_noise_.reset(system_noise.clone());
  input_mapping_ = input_mat;
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state) const {
  return this->propagate(state, Eigen::VectorXd::Zero(input_mapping_.cols()));
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {

  CHECK_EQ(state.size(), system_mapping_.rows());
  CHECK_EQ(input.size(), input_mapping_.cols());

  // If there is no input to the system,
  // we don't need to compute the matrix multiplication.
  if (input_mapping_.size() == 0
      || input == Eigen::VectorXd::Zero(input_mapping_.cols())) {
    return system_mapping_ * state + system_noise_->mean();
  } else {
    return system_mapping_ * state + input_mapping_ * input
        + system_noise_->mean();
  }
}

int LinearSystemModel::getStateDim() const {
  return system_mapping_.rows();
}

int LinearSystemModel::getInputDim() const {
  return input_mapping_.cols();
}

DistributionInterface* LinearSystemModel::getSystemNoise() const {
  return system_noise_.get();
}

Eigen::MatrixXd LinearSystemModel::getJacobian() const {
  return system_mapping_;
}

}  // namespace refill
