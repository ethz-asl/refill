#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <memory>

#include "refill/system_models/system_model_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic, int INPUT_DIM = 0>
class LinearSystemModel : public SystemModelBase<STATE_DIM, INPUT_DIM> {
 public:
  LinearSystemModel();
  LinearSystemModel(
      const Eigen::Matrix<double, STATE_DIM, STATE_DIM>& system_mat,
      const DistributionInterface<STATE_DIM>& system_noise =
          GaussianDistribution<STATE_DIM>(),
      const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>& input_mat =
          Eigen::MatrixXd::Zero(STATE_DIM, INPUT_DIM));

  int GetStateDim() const {
    return system_mat_.rows();
  }

  int GetInputDim() const {
    return input_mat_.cols();
  }

  // Propagate a state vector through the linear system model
  Eigen::Matrix<double, STATE_DIM, 1> Propagate(
      const Eigen::Matrix<double, STATE_DIM, 1>& state,
      const Eigen::Matrix<double, INPUT_DIM, 1>& input = Eigen::VectorXd::Zero(
          INPUT_DIM)) const;

  DistributionInterface<STATE_DIM>* GetSystemNoise() const {
    return system_noise_.get();
  }

  Eigen::Matrix<double, STATE_DIM, STATE_DIM> GetJacobian() const {
    return system_mat_;
  }

 private:
  Eigen::Matrix<double, STATE_DIM, STATE_DIM> system_mat_;
  Eigen::Matrix<double, STATE_DIM, INPUT_DIM> input_mat_;
  std::unique_ptr<DistributionInterface<STATE_DIM>> system_noise_;
};

typedef LinearSystemModel<Eigen::Dynamic, Eigen::Dynamic> LinearSystemModelXd;

}  // namespace refill

#include "./linear_system_model-inl.h"

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
