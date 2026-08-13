#include "activation.hpp"
#include "criterion.hpp"
#include "network.hpp"

#include <print>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>
using namespace axon;

namespace
{
    auto generate_sin_dataset(std::size_t num_samples, double noise_level = 0.05)
        -> std::vector<std::pair<std::vector<double>, std::vector<double>>>
    {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
        dataset.reserve(num_samples);

        std::random_device rd;
        std::mt19937 rng(rd());

        std::uniform_real_distribution<> x_dist(0.0, 2.0 * std::numbers::pi);
        std::normal_distribution<> noise_dist(0.0, noise_level);

        for (std::size_t i = 0; i < num_samples; ++i)
        {
            const double x = x_dist(rng);
            const double x_normalized = x / (2.0 * std::numbers::pi);

            double y = std::clamp(std::sin(x) + noise_dist(rng), -1.0, 1.0);

            dataset.emplace_back(std::vector<double>{x_normalized}, std::vector<double>{y});
        }

        return dataset;
    }
} // namespace

auto main() -> int
{
    std::println("=== Function Approximation: f(x) = sin(x) ===\n");

    const std::vector<std::size_t> topology = {1, 32, 16, 1};

    Activation activation{.function = activation::tanh, .derivative = activation::tanh_derivative};
    Criterion criterion{.function = criterion::mse, .derivative = criterion::mse_derivative};

    Network net(topology, activation, criterion);

    constexpr std::size_t num_train_samples = 1500;
    constexpr double learning_rate = 0.01;
    constexpr double momentum = 0.75;
    constexpr std::size_t num_epochs = 500;

    auto train_dataset = generate_sin_dataset(num_train_samples, 0.002);

    std::println("- Generated {} training samples", train_dataset.size());
    std::println("- Training hyperparameters:");
    std::println("\tLearning rate: {}", learning_rate);
    std::println("\tMomentum: {}", momentum);
    std::println("\tEpochs: {}\n", num_epochs);

    std::random_device rd;
    std::mt19937 rng(rd());

    constexpr int bar_width = 30;
    const int epoch_width = static_cast<int>(std::log10(num_epochs)) + 1;

    for (std::size_t epoch = 0; epoch < num_epochs; ++epoch)
    {
        std::ranges::shuffle(train_dataset, rng);
        double epoch_loss = 0.0;

        for (const auto& [inputs, targets] : train_dataset)
        {
            net.feed_forward(inputs);
            epoch_loss += net.compute_loss(targets);
            net.back_propagate(targets);
            net.step(learning_rate, momentum);
        }

        epoch_loss /= static_cast<double>(train_dataset.size());

        const int filled = static_cast<int>(bar_width * (epoch + 1) / num_epochs);
        const int pct = static_cast<int>(100.0 * (epoch + 1) / num_epochs);

        std::print("\r  \033[36mEpoch\033[0m \033[1m{:>{}}/{}\033[0m [{}", epoch + 1, epoch_width,
                   num_epochs, "\033[32m");

        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
                std::print("█");
            else if (i == filled)
                std::print("\033[33m▸\033[90m");
            else
                std::print("·");
        }

        std::print("\033[0m] \033[33m{:3d}%\033[0m | \033[35mloss\033[0m = {:.6f}  ", pct,
                   epoch_loss);
        std::fflush(stdout);
    }

    std::println("\n=== Testing ===\n");
    std::println("   x      sin(x)   Prediction   Error");
    std::println("-------  --------  ----------  -------");

    double total_error = 0.0;
    for (std::size_t i = 0; i < 20; ++i)
    {
        const double x = (2.0 * std::numbers::pi * static_cast<double>(i)) / 20.0;
        const double x_normalized = x / (2.0 * std::numbers::pi);

        net.feed_forward({x_normalized});
        const auto output = net.get_output();

        const double prediction = output[0];
        const double sin_value = std::sin(x);
        const double error = std::abs(sin_value - prediction);
        total_error += error;

        std::println("{:7.3f}  {:8.3f}  {:11.3f}  {:7.3f}", x, sin_value, prediction, error);
    }

    const double mean_absolute_error = total_error / 20.0;

    std::println("\n- Mean Absolute Error: {:.6f}", mean_absolute_error);

    return 0;
}
