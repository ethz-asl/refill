#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <memory>

#include "refill/system_models/system_model_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template <int STATEDIM = Eigen::Dynamic, int INPUTDIM = Eigen::Dynamic>
class LinearSystemModel : public SystemModelBase<STATEDIM, INPUTDIM> {
 public:
  LinearSystemModel();
  LinearSystemModel(const Eigen::Matrix<double, STATEDIM, STATEDIM>& system_mat,
                    const Eigen::Matrix<double, STATEDIM, INPUTDIM>& input_mat =
                        Eigen::MatrixXd::Zero(STATEDIM, INPUTDIM),
                    const DistributionBase<STATEDIM>& system_noise =
                        new GaussianDistribution<STATEDIM>());

  int dim() const { return system_mat_.cols(); }
  void Propagate(Eigen::Matrix<double, STATEDIM, 1>* state,
                 const Eigen::Matrix<double, INPUTDIM, 1>& input =
                     Eigen::VectorXd::Zero(INPUTDIM));

  DistributionBase<STATEDIM>* GetSystemNoise() { return system_noise_.get(); }

 private:
  Eigen::Matrix<double, STATEDIM, STATEDIM> system_mat_;
  Eigen::Matrix<double, STATEDIM, INPUTDIM> input_mat_;
  std::unique_ptr<DistributionBase<STATEDIM>> system_noise_;
};

using LinearSystemModelXd = LinearSystemModel<Eigen::Dynamic, Eigen::Dynamic>;

}  // namespace refill

#include "./linear_system_model-inl.h"

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
