#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int STATE_DIM, int INPUT_DIM>
class SystemModelBase {
 public:
  virtual void Propagate(Eigen::Matrix<double, STATE_DIM, 1>* state,
                         const Eigen::Matrix<double, INPUT_DIM, 1>& input) = 0;
  virtual int dim() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
