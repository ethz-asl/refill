#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearizedSystemModel : public SystemModelBase {
 public:
  LinearizedSystemModel() {}

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

  int getSystemNoiseDim() const;
 protected:
  std::unique_ptr<DistributionInterface> system_noise_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
