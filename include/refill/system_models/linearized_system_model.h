#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearizedSystemModel : public SystemModelBase {
 public:
  LinearizedSystemModel(const unsigned int& state_dim,
                        const DistributionInterface& system_noise);
  LinearizedSystemModel(const unsigned int& state_dim,
                        const DistributionInterface& system_noise,
                        const unsigned int& input_dim);

  void setLinearizedSystemParameters(const unsigned int& state_dim,
                                     const DistributionInterface& system_noise);
  void setLinearizedSystemParameters(const unsigned int& state_dim,
                                     const DistributionInterface& system_noise,
                                     const unsigned int& input_dim);

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

  unsigned int getStateDim() const;
  unsigned int getSystemNoiseDim() const;
  unsigned int getInputDim() const;
  DistributionInterface* getSystemNoise() const;

 protected:
  std::unique_ptr<DistributionInterface> system_noise_;
  unsigned int state_dim_;
  unsigned int input_dim_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
