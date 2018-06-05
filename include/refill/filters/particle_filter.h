#ifndef REFILL_FILTERS_PARTICLE_FILTER_H_
#define REFILL_FILTERS_PARTICLE_FILTER_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "refill/distributions/distribution_base.h"
#include "refill/distributions/likelihood.h"
#include "refill/filters/filter_base.h"
#include "refill/system_models/system_model_base.h"
#include "refill/utility/resample_methods.h"

namespace refill {

class ParticleFilter : public FilterBase {
 public:
  ParticleFilter();
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist);
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist,
                 const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
                     resample_method);
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist,
                 const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
                     resample_method,
                 std::unique_ptr<SystemModelBase> system_model,
                 std::unique_ptr<Likelihood> measurement_model);
  virtual ~ParticleFilter() = default;

  void setFilterParameters(const std::size_t& n_particles,
                           DistributionInterface* initial_state_dist);
  void setFilterParameters(
      const std::size_t& n_particles, DistributionInterface* initial_state_dist,
      const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
          resample_method);
  void setFilterParameters(
      const std::size_t& n_particles, DistributionInterface* initial_state_dist,
      const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
          resample_method,
      std::unique_ptr<SystemModelBase> system_model,
      std::unique_ptr<Likelihood> measurement_model);

  /** @brief Sets the particles and resets the weights to uniform. */
  void setParticles(const Eigen::MatrixXd& particles);

  /** @brief Sets the particles and corresponding weights. */
  void setParticlesAndWeights(const Eigen::MatrixXd& particles,
                              const Eigen::VectorXd& weights);

  /** @brief Draws particles from the provided distribution
   *         and resets the weights to uniform. */
  void reinitializeParticles(DistributionInterface* initial_state);

  void predict() override;
  void predict(const Eigen::VectorXd& input);
  void predict(const SystemModelBase& system_model);
  void predict(const SystemModelBase& system_model,
               const Eigen::VectorXd& input);

  void update(const Eigen::VectorXd& measurement) override;
  void update(const Likelihood& measurement_model,
              const Eigen::VectorXd& measurement);

  /** @brief Returns the expected value of the current particle distribution. */
  Eigen::VectorXd getExpectation();

  /** @brief Returns the particle with maximum weight. */
  Eigen::VectorXd getMaxWeightParticle();

  /** @brief Returns all particles. */
  Eigen::MatrixXd getParticles();

  /** @brief Returns all particles and corresponding weights. */
  void getParticlesAndWeights(Eigen::MatrixXd* particles,
                              Eigen::VectorXd* weights);

 private:
  std::size_t num_particles_;
  Eigen::MatrixXd particles_;
  Eigen::VectorXd weights_;
  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<Likelihood> measurement_model_;
  std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)> resample_method_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_PARTICLE_FILTER_H_
