#include "refill/filters/filter_base.h"

namespace refill {

FilterBase::FilterBase() {}

FilterBase::FilterBase(std::unique_ptr<SystemModelBase> system_model,
                       std::unique_ptr<MeasurementModelBase> measurement_model)
    : system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)) {}

void FilterBase::predict() {
  CHECK(this->system_model_) << "No default system model provided.";
  this->predict(0);
}

void FilterBase::predict(double stamp) {
  CHECK(this->system_model_) << "No default system model provided.";
  this->predict(stamp, *this->system_model_);
}

void FilterBase::predict(double stamp, SystemModelBase& system_model) {
  const int kInputSize = system_model.getInputDim();
  this->predict(stamp, system_model, Eigen::VectorXd::Zero(kInputSize));
}

void FilterBase::update(const Eigen::VectorXd& measurement) {
  CHECK(this->measurement_model_) << "No default measurement model provided.";
  this->update(*this->measurement_model_, measurement);
}

void FilterBase::update(const MeasurementModelBase& measurement_model,
                        const Eigen::VectorXd& measurement) {
  this->update(measurement_model, measurement, new double());
}
}  // namespace refill