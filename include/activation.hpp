#pragma once

#include <cmath>

namespace axon::activation
{

    // Identity function
    [[nodiscard]] constexpr auto linear(double x) -> double
    {
        return x;
    }

    [[nodiscard]] constexpr auto linear_derivative([[maybe_unused]] double x) -> double
    {
        return 1.0;
    }

    // Logistic function (sigmoid curve)
    [[nodiscard]] constexpr auto sigmoid(double x) -> double
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    [[nodiscard]] constexpr auto sigmoid_derivative(double output) -> double
    {
        return output * (1.0 - output);
    }

    // Hyperbolic tangent function
    [[nodiscard]] constexpr auto tanh(double x) -> double
    {
        return std::tanh(x);
    }

    [[nodiscard]] constexpr auto tanh_derivative(double output) -> double
    {
        return 1.0 - (output * output);
    }

    // Rectified linear unit function
    [[nodiscard]] constexpr auto relu(double x) -> double
    {
        return x > 0.0 ? x : 0.0;
    }

    [[nodiscard]] constexpr auto relu_derivative(double output) -> double
    {
        return output > 0.0 ? 1.0 : 0.0;
    }

} // namespace axon::activation
