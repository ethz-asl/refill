#ifndef INCLUDE_REFILL_FILTERS_FILTER_BASE_H_
#define INCLUDE_REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int STATEDIM = Eigen::Dynamic, int MEASDIM = Eigen::Dynamic>
class FilterBase {
 public:
  virtual void Predict() = 0;
  virtual void Update(Eigen::Matrix<double, MEASDIM, 1> measurement) = 0;
};

}  // namespace refill

#endif  // INCLUDE_REFILL_FILTERS_FILTER_BASE_H_
