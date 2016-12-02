#ifndef REFILL_FILTERS_PARTICLE_FILTER_H_
#define REFILL_FILTERS_PARTICLE_FILTER_H_

#include <glog/logging.h>
#include <Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

#include "refill/distributions/distribution_base.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class ParticleFilter : public FilterBase {
 public:
  ParticleFilter();
  ParticleFilter(const size_t& n_particles,
                 const DistributionInterface& initial_state_dist);
  ParticleFilter(const size_t& n_particles,
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

  struct ParticleType {
    Eigen::VectorXd state;
    double weight;
  };

  std::vector<ParticleType> particles_;
  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<MeasurementModelBase> measurement_model_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_PARTICLE_FILTER_H_
