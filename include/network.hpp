#pragma once

#include "neuron.hpp"

#include <vector>

namespace ann
{

    struct Criterion
    {
        using Function = std::function<double(double, double)>;

        Function function;
        Function derivative;
    };

    class Network
    {
    public:
        explicit Network(const std::vector<std::size_t>& layer_sizes, Activation activation,
                         Criterion criterion);

        [[nodiscard]] auto get_output() const -> std::vector<double>;

        [[nodiscard]] auto get_error() const -> double
        {
            return error_;
        }

        auto feed_forward(const std::vector<double>& inputs) -> void;
        auto compute_loss(const std::vector<double>& targets) -> double;
        auto back_propagate() -> void;

    private:
        std::vector<std::vector<Neuron>> layers_;
        std::vector<double> targets_;
        Activation activation_;
        Criterion criterion_;
        double error_{0.0};
    };

} // namespace ann
