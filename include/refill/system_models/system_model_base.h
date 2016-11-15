#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

template<int STATE_DIM, int INPUT_DIM>
class SystemModelBase {
 public:
  virtual Eigen::Matrix<double, STATE_DIM, 1> Propagate(
      const Eigen::Matrix<double, STATE_DIM, 1>& state,
      const Eigen::Matrix<double, INPUT_DIM, 1>& input) const = 0;
  virtual int GetStateDim() const = 0;
  virtual int GetInputDim() const = 0;
  virtual Eigen::Matrix<double, STATE_DIM, STATE_DIM> GetJacobian() const = 0;
  virtual DistributionInterface<STATE_DIM>* GetSystemNoise() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
