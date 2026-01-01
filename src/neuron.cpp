#include "neuron.hpp"

#include <random>

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
    } // namespace

    Neuron::Neuron(std::size_t num_outputs, std::size_t index, Activation activation)
        : activation_(std::move(activation)),
          index_(index)
    {
        output_weights_.reserve(num_outputs);
        for (std::size_t i{0}; i < num_outputs; ++i)
        {
            output_weights_.emplace_back(get_random_weight());
        }
    }

    auto Neuron::feed_forward(const std::vector<Neuron>& prev_layer) -> void
    {
        double sum{0.0};

        for (const auto& neuron : prev_layer)
        {
            sum += neuron.get_output() * neuron.get_weights()[index_].weight;
        }

        output_value_ = activation_.function(sum);
    }

} // namespace ann
