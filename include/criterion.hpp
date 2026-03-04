#pragma once

namespace axon::criterion
{

    [[nodiscard]] inline auto mse(double target, double output) -> double
    {
        const double error = target - output;
        return error * error;
    }

    [[nodiscard]] inline auto mse_derivative(double target, double output) -> double
    {
        return 2 * (output - target);
    }

} // namespace axon::criterion
