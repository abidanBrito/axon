#include "network.hpp"

#include <stdexcept>

namespace ann
{
    constexpr double bias_constant{1.0};

    Network::Network(const std::vector<std::size_t>& layer_sizes, Activation activation,
                     Criterion criterion)
        : activation_(std::move(activation)),
          criterion_(std::move(criterion))
    {
        if (layer_sizes.empty())
        {
            throw std::invalid_argument("The network can not be empty.");
        }

        layers_.reserve(layer_sizes.size());

        for (std::size_t layer_idx{0}; layer_idx < layer_sizes.size(); ++layer_idx)
        {
            const std::size_t num_neurons = layer_sizes[layer_idx];
            const std::size_t num_outputs =
                (layer_idx == layer_sizes.size() - 1) ? 0 : layer_sizes[layer_idx + 1];

            layers_.emplace_back();
            auto& layer = layers_.back();

            layer.reserve(num_neurons + 1);

            for (std::size_t neuron_idx{0}; neuron_idx < num_neurons; ++neuron_idx)
            {
                if (layer_idx == 0)
                {
                    layer.emplace_back(num_outputs, neuron_idx);
                }
                else
                {
                    layer.emplace_back(num_outputs, neuron_idx, activation_);
                }
            }

            // NOTE(abi): bias neuron has no activation.
            layer.emplace_back(num_outputs, num_neurons);
            layer.back().set_output(bias_constant);
        }
    }

    [[nodiscard]] auto Network::get_output() const -> std::vector<double>
    {
        std::vector<double> output;

        const auto& output_layer = layers_.back();
        output.reserve(output_layer.size() - 1);

        for (std::size_t i{0}; i < output_layer.size() - 1; ++i)
        {
            output.push_back(output_layer[i].get_output());
        }

        return output;
    }

    auto Network::feed_forward(const std::vector<double>& inputs) -> void
    {
        if (inputs.size() != layers_[0].size() - 1)
        {
            throw std::invalid_argument("Invalid number of inputs.");
        }

        for (std::size_t i{0}; i < inputs.size(); ++i)
        {
            layers_[0][i].set_output(inputs[i]);
        }

        // NOTE(abi): we skip the bias neurons from the forward pass.
        for (std::size_t layer_idx{1}; layer_idx < layers_.size(); ++layer_idx)
        {
            const auto& prev_layer = layers_[layer_idx - 1];
            auto& current_layer = layers_[layer_idx];

            for (std::size_t neuron_idx{0}; neuron_idx < current_layer.size() - 1; ++neuron_idx)
            {
                current_layer[neuron_idx].feed_forward(prev_layer);
            }
        }
    }

    auto Network::compute_loss(const std::vector<double>& targets) -> double
    {
        targets_ = targets;
        error_ = 0.0;

        const auto& output_layer = layers_.back();
        for (std::size_t i = 0; i < output_layer.size() - 1; ++i)
        {
            const double target = targets[i];
            const double output = output_layer[i].get_output();
            error_ += criterion_.function(target, output);
        }

        error_ /= static_cast<double>(output_layer.size() - 1);

        return error_;
    }

    auto Network::back_propagate() -> void
    {
        // Output layer gradients
        auto& output_layer = layers_.back();
        for (std::size_t i{0}; i < output_layer.size() - 1; ++i)
        {
            const double target = targets_[i];
            const double output = output_layer[i].get_output();
            const double gradient =
                criterion_.derivative(target, output) * activation_.derivative(output);
            output_layer[i].set_gradient(gradient);
        }

        // Hidden layer gradients
        for (std::size_t layer_idx = layers_.size() - 2; layer_idx > 0; --layer_idx)
        {
            auto& hidden_layer = layers_[layer_idx];
            auto& next_layer = layers_[layer_idx + 1];

            for (auto& neuron : hidden_layer)
            {
                neuron.compute_hidden_gradient(next_layer);
            }
        }
    }

} // namespace ann
