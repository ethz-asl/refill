#include "refill/filters/particle_filter.h"

using std::size_t;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace refill {

ParticleFilter::ParticleFilter()
    : num_particles_(0),
      particles_(0, 0),
      weights_(0),
      likelihood_(nullptr),
      resample_method_(nullptr) {}

ParticleFilter::ParticleFilter(const size_t& n_particles,
                               DistributionInterface* initial_state_dist)
    : num_particles_(n_particles),
      particles_(initial_state_dist->mean().rows(), n_particles),
      weights_(n_particles),
      likelihood_(nullptr),
      resample_method_(nullptr) {
  reinitializeParticles(initial_state_dist);
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method)
    : num_particles_(n_particles),
      particles_(initial_state_dist->mean().rows(), n_particles),
      weights_(n_particles),
      likelihood_(nullptr),
      resample_method_(resample_method) {
  reinitializeParticles(initial_state_dist);
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<Likelihood> likelihood)
    : num_particles_(n_particles),
      particles_(initial_state_dist->mean().rows(), n_particles),
      weights_(n_particles),
      likelihood_(std::move(likelihood)),
      resample_method_(resample_method),
      FilterBase(std::move(system_model), nullptr) {
  reinitializeParticles(initial_state_dist);
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, DistributionInterface* initial_state_dist) {
  num_particles_ = n_particles;
  this->reinitializeParticles(initial_state_dist);
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method) {
  this->setFilterParameters(n_particles, initial_state_dist);

  resample_method_ = resample_method;
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<Likelihood> measurement_model) {
  this->setFilterParameters(n_particles, initial_state_dist, resample_method);

  system_model_ = std::move(system_model);
  likelihood_ = std::move(measurement_model);
}

void ParticleFilter::setParticles(const Eigen::MatrixXd& particles) {
  CHECK_EQ(particles_.rows(), particles.rows());
  CHECK_EQ(particles_.cols(), particles.cols());

  particles_ = particles;
  weights_.setConstant(1.0 / num_particles_);
}

void ParticleFilter::setParticlesAndWeights(const Eigen::MatrixXd& particles,
                                            const Eigen::VectorXd& weights) {
  CHECK_EQ(particles_.rows(), particles.rows());
  CHECK_EQ(weights.rows(), particles.cols());
  CHECK_NE(weights.squaredNorm(), 0.0) << "Zero weight vector is not allowed!";

  particles_ = particles;
  weights_ = weights / weights.sum();
}

void ParticleFilter::reinitializeParticles(
    DistributionInterface* initial_state_dist) {
  particles_.resize(initial_state_dist->mean().rows(), num_particles_);
  weights_.resize(num_particles_);

  weights_.setConstant(1.0 / num_particles_);
  for (int i = 0; i < num_particles_; ++i) {
    particles_.col(i) = initial_state_dist->drawSample();
  }
}

void ParticleFilter::predict(const double dt, SystemModelBase& system_model,
                             const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getInputDim(), input.rows());
  CHECK_EQ(system_model.getStateDim(), particles_.rows());
  CHECK_NE(particles_.cols(), 0) << "Particle vector is empty.";
  CHECK(dt >= 0) << "Negative dt: Cannot perform prediction!";

  for (int i = 0; i < num_particles_; ++i) {
    Eigen::VectorXd noise_sample = system_model.getNoise()->drawSample();
    particles_.col(i) =
        system_model.propagate(particles_.col(i), input, noise_sample);
  }
}

void ParticleFilter::update(const MeasurementModelBase& measurement_model,
                            const Eigen::VectorXd& measurement, double* likelihood) {
  Eigen::VectorXd likelihoods =
      likelihood_->getLikelihoodVectorized(particles_, measurement);

  weights_ = weights_.cwiseProduct(likelihoods);

  weights_ /= weights_.sum();

  // Resample using the provided resampling method
  if (resample_method_) {
    resample_method_(&particles_, &weights_);
  }
}

Eigen::VectorXd ParticleFilter::getExpectation() {
  return particles_ * weights_;
}

Eigen::VectorXd ParticleFilter::getMaxWeightParticle() {
  Eigen::VectorXd::Index index;

  double max_weight = weights_.maxCoeff(&index);

  return particles_.col(index);
}

Eigen::MatrixXd ParticleFilter::getParticles() { return particles_; }

void ParticleFilter::getParticlesAndWeights(Eigen::MatrixXd* particles,
                                            Eigen::VectorXd* weights) {
  *particles = particles_;
  *weights = weights_;
}

}  // namespace refill
