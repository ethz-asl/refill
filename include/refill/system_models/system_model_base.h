#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <memory>

#include "refill/distributions/distribution_base.h"

namespace refill {

class SystemModelBase {
 public:
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input) const = 0;

  std::size_t getStateDim() const;
  std::size_t getInputDim() const;
  std::size_t getSystemNoiseDim() const;
  DistributionInterface* getSystemNoise() const;

 protected:
  SystemModelBase() = delete;
  SystemModelBase(const std::size_t& state_dim,
                  const DistributionInterface& system_noise);
  SystemModelBase(const std::size_t& state_dim,
                  const DistributionInterface& system_noise,
                  const std::size_t& input_dim);

  void setSystemModelBaseParameters(const std::size_t& state_dim,
                           const DistributionInterface& system_noise);
  void setSystemModelBaseParameters(const std::size_t& state_dim,
                           const DistributionInterface& system_noise,
                           const std::size_t& input_dim);

 private:
  std::size_t state_dim_;
  std::size_t input_dim_;
  std::unique_ptr<DistributionInterface> system_noise_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
