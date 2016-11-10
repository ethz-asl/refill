#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int STATEDIM, int INPUTDIM>
class SystemModelBase {
 public:
  virtual void Propagate(Eigen::Matrix<double, STATEDIM, 1>* state,
                         const Eigen::Matrix<double, INPUTDIM, 1>& input) = 0;
  virtual int dim() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
