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

typedef LinearSystemModel<Eigen::Dynamic, Eigen::Dynamic>
        LinearSystemModelXd;

// Function definitions

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel() {
  constexpr int kCurrentSystemDIm =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;
  constexpr int kCurrentInputDIm =
      (INPUT_DIM == Eigen::Dynamic) ? 1 : INPUT_DIM;

  system_mat_ =
      Eigen::Matrix<double, kCurrentSystemDIm, kCurrentSystemDIm>::Identity(
          kCurrentSystemDIm, kCurrentSystemDIm);
  input_mat_ =
      Eigen::Matrix<double, kCurrentSystemDIm, kCurrentInputDIm>::Identity(
          kCurrentSystemDIm, kCurrentInputDIm);

  // In case of no declaration of system noise, we use standard normal gaussian.
  system_noise_.reset(new GaussianDistribution<kCurrentSystemDIm>());
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise)
    : system_mat_(system_mat),
      system_noise_(system_noise.clone()),
      input_mat_(Eigen::MatrixXd::Zero(0, 0)) {
  const int kStateDim = system_mat.rows();

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(kStateDim, system_mat.cols());
    CHECK_EQ(kStateDim, system_noise.mean().size());
  }
}

template<int STATE_DIM, int INPUT_DIM>
LinearSystemModel<STATE_DIM, INPUT_DIM>::LinearSystemModel(
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
    const DistributionInterface<STATE_DIM>& system_noise,
    const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat)
    : system_mat_(system_mat),
      input_mat_(input_mat),
      system_noise_(system_noise.clone()) {
  const int kStateDim = system_mat.rows();

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(kStateDim, system_mat.cols());
    CHECK_EQ(kStateDim, input_mat.rows());
    CHECK_EQ(kStateDim, system_noise.mean().size());
  }
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
    const Eigen::Matrix<double, INPUT_DIM, 1>& input) const {

  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state.size(), system_mat_.rows());
  }

  if (INPUT_DIM == Eigen::Dynamic) {
    CHECK_EQ(input.rows(), input_mat_.cols());
  }

  // If there is no input, we don't need to compute the matrix multiplication.
  if (input_mat_.cols() == 0) {
    return system_mat_ * state + system_noise_->mean();
  } else {
    return system_mat_ * state + input_mat_ * input + system_noise_->mean();
  }
}

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
