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
  unsigned int sample_set_type = kUkfUseMultipleSampleSets;
  unsigned int mat_square_root_algo = kUkfUseCholeskyDecomposition;
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

  void predict() override;
  void predict(const Eigen::VectorXd& input);
  void predict(const SystemModelBase& system_model);
  void predict(const SystemModelBase& system_model,
               const Eigen::VectorXd& input);
  void update(const Eigen::VectorXd& measurement) override;
  void update(const MeasurementModelBase& measurement_model,
              const Eigen::VectorXd& measurement);

  void setUkfSettings(UkfSettings settings);

 private:
  GaussianDistribution state_;
  UkfSettings settings_;

  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<MeasurementModelBase> measurement_model_;

  // Only state_samples variable is used when the joint distribution is sampled.
  // In that case the noise_samples variable remains untouched.
  Eigen::MatrixXd state_samples_;
  Eigen::MatrixXd noise_samples_;

  // Computes unscented transform of the current system state and noise.
  void computeUnscentedTransform(
      const GaussianDistribution& noise_distribution);
};

}  // namespace refill

#endif  // REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H_
