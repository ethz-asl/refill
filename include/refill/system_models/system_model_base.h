#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

class SystemModelBase {
 public:
  virtual Eigen::VectorXd propagate(
      const Eigen::VectorXd& state,
      const Eigen::VectorXd& input) const = 0;
  virtual int getStateDim() const = 0;
  virtual int getInputDim() const = 0;
  virtual DistributionInterface* getSystemNoise() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
