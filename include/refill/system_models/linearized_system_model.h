#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>

#include <memory>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearizedSystemModel : public SystemModelBase {
 public:
  LinearizedSystemModel() = delete;
  LinearizedSystemModel(const int& system_dim,
                        const DistributionInterface& system_noise);
  LinearizedSystemModel(const int& system_dim,
                        const DistributionInterface& system_noise,
                        const int& input_dim);

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

  int getSystemNoiseDim() const;
  DistributionInterface* getSystemNoise() const;
 protected:
  std::unique_ptr<DistributionInterface> system_noise_;
  const int system_dim_;
  const int input_dim_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
