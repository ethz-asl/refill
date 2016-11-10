#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

template<int STATEDIM, int INPUTDIM>
class SystemModelBase {
 public:
  virtual Eigen::Matrix<double, STATEDIM, 1> Propagate(
      const Eigen::Matrix<double, STATEDIM, 1>& state,
      const Eigen::Matrix<double, INPUTDIM, 1>& input) const = 0;
  virtual int dim() const = 0;
  virtual DistributionInterface<STATEDIM>* GetSystemNoise() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
