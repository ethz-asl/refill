#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearizedSystemModel : public SystemModelBase {
 public:
  LinearizedSystemModel(const std::size_t& state_dim,
                        const DistributionInterface& system_noise);
  LinearizedSystemModel(const std::size_t& state_dim,
                        const DistributionInterface& system_noise,
                        const std::size_t& input_dim);

  void setLinearizedSystemParameters(const std::size_t& state_dim,
                                     const DistributionInterface& system_noise);
  void setLinearizedSystemParameters(const std::size_t& state_dim,
                                     const DistributionInterface& system_noise,
                                     const std::size_t& input_dim);

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

  std::size_t getStateDim() const;
  std::size_t getSystemNoiseDim() const;
  std::size_t getInputDim() const;
  DistributionInterface* getSystemNoise() const;

 protected:
  std::unique_ptr<DistributionInterface> system_noise_;
  std::size_t state_dim_;
  std::size_t input_dim_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
