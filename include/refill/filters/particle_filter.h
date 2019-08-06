#ifndef REFILL_FILTERS_PARTICLE_FILTER_H_
#define REFILL_FILTERS_PARTICLE_FILTER_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include <functional>
#include <memory>

#include "refill/distributions/distribution_base.h"
#include "refill/distributions/likelihood.h"
#include "refill/filters/filter_base.h"
#include "refill/system_models/system_model_base.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/utility/resample_methods.h"

namespace refill {

class ParticleFilter : public FilterBase {
 public:
  /** @brief Constructs the particle filter with no value. */
  ParticleFilter();
  /**
   * @brief Constructs a new particle filter so that a system model is expected
   *        to be given upon prediction / update and no resampling will be
   *        performed. */
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist);
  /**
   * @brief Constructs a new particle filter so that a system model is expected
   *        to be given upon prediction / update and resampling is performed. */
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist,
                 const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
                     resample_method);
  /**
   * @brief Constructs a new particle filter which performes resampling and
   *        uses the standard models, if not stated otherwise. */
  ParticleFilter(const std::size_t& n_particles,
                 DistributionInterface* initial_state_dist,
                 const std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)>&
                     resample_method,
                 std::unique_ptr<SystemModelBase> system_model,
                 std::unique_ptr<Likelihood> measurement_model);
  /**
   * @brief Destroy the Particle Filter object using default destructor. */
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

  using FilterBase::predict;
  void predict(const double stamp, SystemModelBase& system_model,
               const Eigen::VectorXd& input) override;

  using FilterBase::update;
  void update(const MeasurementModelBase& measurement_model,
              const Eigen::VectorXd& measurement, double* likelihood) override;

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
  std::unique_ptr<Likelihood> likelihood_;
  std::function<void(Eigen::MatrixXd*, Eigen::VectorXd*)> resample_method_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_PARTICLE_FILTER_H_
