#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int STATE_DIM = Eigen::Dynamic, int MEAS_DIM = Eigen::Dynamic>
class FilterBase {
 public:
  virtual void Predict() = 0;
  virtual void Update(Eigen::Matrix<double, MEAS_DIM, 1> measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
