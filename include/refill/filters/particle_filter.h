#ifndef REFILL_FILTERS_PARTICLE_FILTER_H_
#define REFILL_FILTERS_PARTICLE_FILTER_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "refill/distributions/distribution_base.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

template<typename ParticleType>
class ParticleFilter : public FilterBase {
 public:
  using ParticleVectorType = std::vector<std::pair<ParticleType, double>>;

  ParticleFilter();
  ParticleFilter(const std::size_t& n_particles,
                 const DistributionInterface& initial_state_dist);
  ParticleFilter(const std::size_t& n_particles,
                 const DistributionInterface& initial_state_dist,
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

 private:
  void resample();

  ParticleVectorType particles_;
  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<MeasurementModelBase> measurement_model_;
  std::function<void(ParticleVectorType*)> resample_method_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_PARTICLE_FILTER_H_
