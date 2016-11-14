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
  LinearSystemModel(const Eigen::Matrix<double, STATE_DIM, STATE_DIM>&
                      system_mat,
                    const DistributionInterface<STATE_DIM>& system_noise =
                        GaussianDistribution<STATE_DIM>(),
                    const Eigen::Matrix<double, STATE_DIM, INPUT_DIM>&
                      input_mat =
                        Eigen::MatrixXd::Zero(STATE_DIM, INPUT_DIM));

  int dim() const { return system_mat_.cols(); }
  void Propagate(Eigen::Matrix<double, STATE_DIM, 1>* state,
                 const Eigen::Matrix<double, INPUT_DIM, 1>& input =
                     Eigen::VectorXd::Zero(INPUT_DIM));

  DistributionInterface<STATE_DIM>* GetSystemNoise() {
    return system_noise_.get();
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
