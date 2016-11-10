#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <memory>

#include "refill/system_models/system_model_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template<int STATEDIM = Eigen::Dynamic, int INPUTDIM = 0>
class LinearSystemModel : public SystemModelBase<STATEDIM, INPUTDIM> {
 public:
  LinearSystemModel();
  LinearSystemModel(const Eigen::Matrix<double, STATEDIM, STATEDIM>& system_mat,
                    const DistributionInterface<STATEDIM>& system_noise =
                        GaussianDistribution<STATEDIM>(),
                    const Eigen::Matrix<double, STATEDIM, INPUTDIM>& input_mat =
                        Eigen::MatrixXd::Zero(STATEDIM, INPUTDIM));

  int dim() const { return system_mat_.cols(); }
  void Propagate(Eigen::Matrix<double, STATEDIM, 1>* state,
                 const Eigen::Matrix<double, INPUTDIM, 1>& input =
                     Eigen::VectorXd::Zero(INPUTDIM));

  DistributionInterface<STATEDIM>* GetSystemNoise() {
    return system_noise_.get();
  }

 private:
  Eigen::Matrix<double, STATEDIM, STATEDIM> system_mat_;
  Eigen::Matrix<double, STATEDIM, INPUTDIM> input_mat_;
  std::unique_ptr<DistributionInterface<STATEDIM>> system_noise_;
};

typedef LinearSystemModel<Eigen::Dynamic, Eigen::Dynamic> LinearSystemModelXd;

}  // namespace refill

#include "./linear_system_model-inl.h"

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
