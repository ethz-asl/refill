#include "refill/filters/particle_filter.h"

using std::size_t;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace refill {

ParticleFilter::ParticleFilter()
    : num_particles_(0),
      particles_(0, 0),
      system_model_(nullptr),
      measurement_model_(nullptr),
      resample_method_(nullptr) {
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method)
    : num_particles_(n_particles),
      particles_(initial_state_dist->mean().rows(), n_particles),
      system_model_(nullptr),
      measurement_model_(nullptr),
      resample_method_(resample_method) {
  initializeParticles(initial_state_dist);
}

ParticleFilter::ParticleFilter(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<Likelihood> measurement_model)
    : num_particles_(n_particles),
      particles_(initial_state_dist->mean().rows(), n_particles),
      system_model_(std::move(system_model)),
      measurement_model_(std::move(measurement_model)),
      resample_method_(resample_method) {
  initializeParticles(initial_state_dist);
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method) {
  num_particles_ = n_particles;
  initializeParticles(initial_state_dist);

  resample_method_ = resample_method;
}

void ParticleFilter::setFilterParameters(
    const size_t& n_particles, DistributionInterface* initial_state_dist,
    const std::function<void(MatrixXd*, VectorXd*)>& resample_method,
    std::unique_ptr<SystemModelBase> system_model,
    std::unique_ptr<Likelihood> measurement_model) {
  this->setFilterParameters(n_particles, initial_state_dist, resample_method);

  system_model_ = std::move(system_model);
  measurement_model_ = std::move(measurement_model);
}

void ParticleFilter::initializeParticles(
    DistributionInterface* initial_state_dist) {
  particles_.resize(initial_state_dist->mean().rows(), num_particles_);
  weights_.resize(num_particles_);

  double equal_weight = 1 / particles_.cols();
  weights_.setConstant(equal_weight);
  for (int i = 0; i < num_particles_; ++i) {
    particles_.col(i) = initial_state_dist->drawSample();
  }
}

void ParticleFilter::predict() {
  this->predict(Eigen::VectorXd::Zero(system_model_->getInputDim()));
}

void ParticleFilter::predict(const Eigen::VectorXd& input) {
  this->predict(*system_model_, input);
}

void ParticleFilter::predict(const SystemModelBase& system_model) {
  this->predict(system_model,
                Eigen::VectorXd::Zero(system_model.getInputDim()));
}

void ParticleFilter::predict(const SystemModelBase& system_model,
                             const Eigen::VectorXd& input) {
  CHECK_EQ(system_model.getInputDim(), input.rows());
  CHECK_NE(particles_.cols(), 0)<< "Particle vector is empty.";

  for (int i = 0; i < num_particles_; ++i) {
    Eigen::VectorXd noise_sample = system_model.getNoise()->drawSample();
    particles_.col(i) = system_model.propagate(particles_.col(i), input,
                                               noise_sample);
  }
}

void ParticleFilter::update(const Eigen::VectorXd& measurement) {
  this->update(*measurement_model_, measurement);
}

void ParticleFilter::update(const Likelihood& measurement_model,
                            const Eigen::VectorXd& measurement) {
  Eigen::VectorXd likelihoods = measurement_model.getLikelihoodVectorized(
      particles_, measurement);

  weights_ = weights_.cwiseProduct(likelihoods);

  weights_ /= weights_.sum();

  resample_method_(&particles_, &weights_);
}

}  // namespace refill
