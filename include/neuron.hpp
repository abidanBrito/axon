#pragma once

#include "connection.hpp"

#include <vector>
#include <functional>
#include <optional>

namespace ann
{

    struct Activation
    {
        using Function = std::function<double(double)>;

        Function function;
        Function derivative;
    };

    class Neuron
    {
    public:
        Neuron(std::size_t num_outputs, std::size_t index,
               std::optional<Activation> activation = std::nullopt);

        auto set_output(double value) -> void
        {
            output_value_ = value;
        }

        auto set_gradient(double value) -> void
        {
            gradient_ = value;
        }

        [[nodiscard]] auto get_output() const -> double
        {
            return output_value_;
        }

        [[nodiscard]] auto get_gradient() const -> double
        {
            return gradient_;
        }

        [[nodiscard]] auto get_connections() const -> const std::vector<Connection>&
        {
            return connections_;
        }

        auto feed_forward(const std::vector<Neuron>& prev_layer) -> void;
        auto compute_hidden_gradient(std::vector<Neuron>& next_layer) -> void;

    private:
        double output_value_{0.0};
        double gradient_{0.0};
        std::vector<Connection> connections_;
        std::optional<Activation> activation_;
        std::size_t index_;
    };

} // namespace ann
