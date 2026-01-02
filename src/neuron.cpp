#include "neuron.hpp"

#include <random>
#include <stdexcept>

namespace ann
{

    namespace
    {
        auto get_random_weight() -> double
        {
            static std::random_device rd;
            static std::mt19937 prng(rd());
            static std::uniform_real_distribution<> distribution(-1.0, 1.0);

            return distribution(prng);
        }

        auto sum_weighted_gradients(const std::vector<Connection>& connections,
                                    const std::vector<Neuron>& next_layer) -> double
        {
            double sum{0.0};
            for (std::size_t i{0}; i < next_layer.size() - 1; ++i)
            {
                sum += connections[i].weight * next_layer[i].get_gradient();
            }

            return sum;
        }

    } // namespace

    Neuron::Neuron(std::size_t num_outputs, std::size_t index, std::optional<Activation> activation)
        : activation_(std::move(activation)),
          index_(index)
    {
        connections_.reserve(num_outputs);
        for (std::size_t i{0}; i < num_outputs; ++i)
        {
            connections_.emplace_back(get_random_weight());
        }
    }

    auto Neuron::feed_forward(const std::vector<Neuron>& prev_layer) -> void
    {
        if (!activation_)
        {
            throw std::logic_error("Input neurons can't perform a forward pass.");
        }

        double sum{0.0};
        for (const auto& neuron : prev_layer)
        {
            sum += neuron.get_output() * neuron.get_connections()[index_].weight;
        }

        output_value_ = activation_->function(sum);
    }

    auto Neuron::compute_hidden_gradient(std::vector<Neuron>& next_layer) -> void
    {
        const double dow = sum_weighted_gradients(connections_, next_layer);
        gradient_ = dow * activation_->derivative(output_value_);
    }

} // namespace ann
