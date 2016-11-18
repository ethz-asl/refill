#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

class LinearMeasurementModel : public LinearizedMeasurementModel {
 public:
  LinearMeasurementModel();
  LinearMeasurementModel(const Eigen::MatrixXd& measurement_mapping,
                         const DistributionInterface& measurement_noise);
  LinearMeasurementModel(const Eigen::MatrixXd& measurement_mapping,
                         const DistributionInterface& measurement_noise,
                         const Eigen::MatrixXd& noise_mapping);

  void setMeasurementParameters(const Eigen::MatrixXd& measurement_mapping,
                                const DistributionInterface& measurement_noise);
  void setMeasurementParameters(const Eigen::MatrixXd& measurement_mapping,
                                const DistributionInterface& measurement_noise,
                                const Eigen::MatrixXd& noise_mapping);

  Eigen::VectorXd observe(const Eigen::VectorXd& state) const;

  int getStateDim() const;
  int getMeasurementDim() const;
  Eigen::MatrixXd getMeasurementJacobian(const Eigen::VectorXd& state) const;
  Eigen::MatrixXd getNoiseJacobian(const Eigen::VectorXd& state) const;
  DistributionInterface* getMeasurementNoise() const;

 private:
  Eigen::MatrixXd measurement_mapping_;
  Eigen::MatrixXd noise_mapping_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
