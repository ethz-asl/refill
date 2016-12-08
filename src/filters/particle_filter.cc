#include "refill/filters/particle_filter.h"

using std::size_t;

namespace refill {

ParticleFilter::ParticleFilter()
    : particles_(0),
      system_model_(nullptr),
      measurement_model_(nullptr),
      resample_method_(nullptr) {
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, const DistributionInterface& initial_state_dist,
    const std::function<void(std::vector<ParticleType>*)>& resample_method)
    : particles_(n_particles),
      system_model_(nullptr),
      measurement_model_(nullptr),
      resample_method_(resample_method) {
  initializeParticles(initial_state_dist);
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, const DistributionInterface& initial_state_dist,
    const std::function<void(std::vector<ParticleType>*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<MeasurementModelBase> measurement_model)
    : particles_(n_particles),
      system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)),
      resample_method_(resample_method) {
  initializeParticles(initial_state_dist);
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, const DistributionInterface& initial_state_dist,
    const std::function<void(std::vector<ParticleType>*)>& resample_method) {
  particles_.resize(n_particles);
  initializeParticles(initial_state_dist);

  resample_method_ = resample_method;
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, const DistributionInterface& initial_state_dist,
    const std::function<void(std::vector<ParticleType>*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<MeasurementModelBase> measurement_model) {
  particles_.resize(n_particles);
  initializeParticles(initial_state_dist);

  system_model_ = std::move(system_model);
  measurement_model_ = std::move(measurement_model);

  resample_method_ = resample_method;
}

void ParticleFilter::initializeParticles(
    const DistributionInterface& initial_state_dist) {
  CHECK_NE(particles_.size(), 0) << "Particle vector is empty.";

  double equal_weight = 1 / particles_.size();
  for (ParticleType& particle : particles_) {
    particle.first = initial_state_dist.drawSample();
    particle.second = equal_weight;
  }
}

void ParticleFilter::predict() {
  this->predict(Eigen::VectorXd::Zeros(system_model_->getInputDim()));
}

void ParticleFilter::predict(const Eigen::VectorXd& input) {
  CHECK_NE(particles_.size(), 0) << "Particle vector is empty.";

  for (ParticleType& particle : particles_) {
    particle.first = system_model_->propagate(particle.first, input);
  }
}

void ParticleFilter::predict(const SystemModelBase& system_model) {
  this->predict(system_model,
                Eigen::VectorXd::Zeros(system_model.getInputDim()));
}

void ParticleFilter::predict(const SystemModelBase& system_model,
                            const Eigen::VectorXd& input) {
  CHECK_NE(particles_.size(), 0) << "Particle vector is empty.";

  for (ParticleType& particle : particles_) {
    particle.first = system_model.propagate(particle.first, input);
  }
}

void ParticleFilter::update(const Eigen::VectorXd& measurement) {
}

void ParticleFilter::resample() {
  resample_method_(&particles_);
}

}  // namespace refill
