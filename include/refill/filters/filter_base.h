#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

/**
 * @brief Interface for all filters
 *
 * This class is an interface for all filters in refill.
 *
 * It must be an ancestor of all filters implemented in refill.
 */
class FilterBase {
 public:
  /** @brief Perform a prediction step. */
  virtual void predict() = 0;
  /** @brief Perform an update with a measurement. */
  virtual void update(const Eigen::VectorXd& measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
