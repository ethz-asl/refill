#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <glog/logging.h>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearSystemModel : public SystemModelBase {
 public:
  LinearSystemModel();
  LinearSystemModel(const Eigen::MatrixXd& system_mat,
                    const DistributionInterface& system_noise);
  LinearSystemModel(const Eigen::MatrixXd& system_mat,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mat);

  void setSystemParameters(const Eigen::MatrixXd& system_mat,
                           const DistributionInterface& system_noise);
  void setSystemParameters(const Eigen::MatrixXd& system_mat,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mat);
  int getStateDim() const {
    return system_mat_.rows();
  }

  int getInputDim() const {
    return input_mat_.cols();
  }

  // Propagate a state vector through the linear system model
  Eigen::VectorXd propagate(const Eigen::VectorXd& state) const;
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input) const;

  DistributionInterface* getSystemNoise() const {
    CHECK_NE(system_noise_.get(), static_cast<DistributionInterface*>(nullptr));
    return system_noise_.get();
  }

  Eigen::MatrixXd getJacobian() const {
    return system_mat_;
  }

 private:
  Eigen::MatrixXd system_mat_;
  Eigen::MatrixXd input_mat_;
  std::unique_ptr<DistributionInterface> system_noise_;
};

// Function definitions

LinearSystemModel::LinearSystemModel() {
  // If standard constructor is called, we assume a one dimensional system
  // without input.
  system_mat_ = Eigen::MatrixXd::Identity(1, 1);
  input_mat_ = Eigen::MatrixXd::Zero(0, 0);

  system_noise_.reset(new GaussianDistribution(1));
}

// If constructor is called without input matrix, we assume there is no input
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mat,
                                     const DistributionInterface& system_noise)
    : system_mat_(system_mat),
      system_noise_(system_noise.clone()),
      input_mat_(Eigen::MatrixXd::Zero(0, 0)) {
  CHECK_EQ(system_mat.rows(), system_mat.cols());
  CHECK_EQ(system_mat.rows(), system_noise.mean().size());
}

LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mat,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
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

  if (input_mat.cols() != 0) {
    CHECK_EQ(system_mat.rows(), input_mat.rows());
  }
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state) const {
  return this->propagate(state, Eigen::VectorXd::Zero(input_mat_.cols()));
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {

  CHECK_EQ(state.size(), system_mat_.rows());
  CHECK_EQ(input.size(), input_mat_.cols());

  // If there is no input to the system,
  // we don't need to compute the matrix multiplication.
  if (input_mat_.cols() == 0 || input_mat_.rows() == 0
      || input == Eigen::VectorXd::Zero(input_mat_.cols())) {
    return system_mat_ * state + system_noise_->mean();
  } else {
    return system_mat_ * state + input_mat_ * input + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
