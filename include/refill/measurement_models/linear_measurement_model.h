#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <glog/logging.h>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/linearizable_measurement_model.h"

namespace refill {

class LinearMeasurementModel : public LinearizableMeasurementModel {
 public:
  LinearMeasurementModel();
  explicit LinearMeasurementModel(const Eigen::MatrixXd& measurement_mat);
  LinearMeasurementModel(const Eigen::MatrixXd& measurement_mat,
                         const DistributionInterface& measurement_noise);

  void setMeasurementParameters(const Eigen::MatrixXd& measurement_mat,
                                const DistributionInterface& measurement_noise);

  Eigen::VectorXd observe(const Eigen::VectorXd& state) const;

  int getStateDim() const;
  int getMeasurementDim() const;
  Eigen::MatrixXd getJacobian() const;
  DistributionInterface* getMeasurementNoise() const;

 private:
  // TODO(jwidauer): Implement noise matrix
  Eigen::MatrixXd measurement_mapping_;
  std::unique_ptr<DistributionInterface> measurement_noise_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
