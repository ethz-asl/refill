#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <glog/logging.h>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic, int INPUT_DIM = 0>
class LinearSystemModel : public SystemModelBase<STATE_DIM, INPUT_DIM> {
 public:
  LinearSystemModel();
  LinearSystemModel(
      const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
      const DistributionInterface<STATE_DIM>& system_noise);
  LinearSystemModel(
      const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
      const DistributionInterface<STATE_DIM>& system_noise,
      const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat);

  int getStateDim() const {
    return system_mat_.rows();
  }

  int getInputDim() const {
    return input_mat_.cols();
  }

  // Propagate a state vector through the linear system model
  Eigen::Matrix<double, STATE_DIM, 1> propagate(
      const Eigen::Matrix<double, STATE_DIM, 1>& state) const;
  Eigen::Matrix<double, STATE_DIM, 1> propagate(
      const Eigen::Matrix<double, STATE_DIM, 1>& state,
      const Eigen::Matrix<double, INPUT_DIM, 1>& input) const;

  DistributionInterface<STATE_DIM>* getSystemNoise() const {
    CHECK_NE(system_noise_.get(),
             static_cast<DistributionInterface<STATE_DIM>*>(nullptr));
    return system_noise_.get();
  }

  Eigen::Matrix<double, STATE_DIM, STATE_DIM> getJacobian() const {
    return system_mat_;
  }

 private:
  Eigen::Matrix<double, STATE_DIM, STATE_DIM> system_mat_;
  Eigen::Matrix<double, STATE_DIM, INPUT_DIM> input_mat_;
  std::unique_ptr<DistributionInterface<STATE_DIM>> system_noise_;
};

typedef LinearSystemModel<Eigen::Dynamic, Eigen::Dynamic> LinearSystemModelXd;

// Function definitions

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel() {
  constexpr int kCurrentSystemDim =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;
  constexpr int kCurrentInputDim =
      (INPUT_DIM == Eigen::Dynamic) ? 1 : INPUT_DIM;

  system_mat_ =
      Eigen::Matrix<double, kCurrentSystemDim, kCurrentSystemDim>::Identity(
          kCurrentSystemDim, kCurrentSystemDim);
  input_mat_ =
      Eigen::Matrix<double, kCurrentSystemDim, kCurrentInputDim>::Identity(
          kCurrentSystemDim, kCurrentInputDim);

  // In case of no declaration of system noise, we use standard normal gaussian
  system_noise_.reset(new GaussianDistribution<kCurrentSystemDim>());
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise)
    : system_mat_(system_mat),
      system_noise_(system_noise.clone()),
      input_mat_(Eigen::MatrixXd::Zero(STATE_DIM, INPUT_DIM)) {
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise,
    const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.clone()) {
}

template<int STATE_DIM, int INPUT_DIM>
Eigen::Matrix<double, STATE_DIM, 1>
  LinearSystemModel<STATE_DIM, INPUT_DIM>::propagate(
    const Eigen::Matrix<double, STATE_DIM, 1>& state) const {
  return this->propagate(state, Eigen::VectorXd::Zero(INPUT_DIM));
}

template<int STATE_DIM, int INPUT_DIM>
Eigen::Matrix<double, STATE_DIM, 1>
  LinearSystemModel<STATE_DIM, INPUT_DIM>::propagate(
    const Eigen::Matrix<double, STATE_DIM, 1>& state,
    const Eigen::Matrix<double, INPUT_DIM, 1> &input) const {

  // If there is no input, we don't need to compute the matrix multiplication.
  if (INPUT_DIM == 0) {
    *state = system_mat_ * (*state) + system_noise_->mean();
  } else {
    *state = system_mat_ * (*state) + input_mat_ * input
        + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
