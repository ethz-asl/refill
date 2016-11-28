#ifndef REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H_
#define REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H_

#include <refill/filters/filter_base.h>

#include <Eigen/Dense>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

constexpr unsigned int kUkfUseMultipleSampleSets = 1u;
constexpr unsigned int kUkfSampleJointDistribution = 2u;

constexpr unsigned int kUkfUseEigendecomposition = 1u;
constexpr unsigned int kUkfUseCholeskyDecomposition = 2u;

typedef struct {
  const unsigned int sample_set_type = kUkfUseMultipleSampleSets;
  const unsigned int mat_square_root_algo = kUkfUseCholeskyDecomposition;
} UkfSettings;

class UnscentedKalmanFilter : public FilterBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  UnscentedKalmanFilter();
  explicit UnscentedKalmanFilter(const GaussianDistribution& initial_state);
  UnscentedKalmanFilter(
      const GaussianDistribution& initial_state,
      std::unique_ptr<SystemModelBase> system_model,
      std::unique_ptr<MeasurementModelBase> measurement_model);

  void predict();
  void update(const Eigen::VectorXd& measurement);

  void computeUnscentedTransform();

 private:
  GaussianDistribution state_;
  Eigen::MatrixXd deterministic_samples;
  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<MeasurementModelBase> measurement_model_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H_
