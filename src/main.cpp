#include "network.hpp"
#include "activation.hpp"
#include "criterion.hpp"

#include <print>

using namespace ann;

auto main() -> int
{
    Activation activation{.function = activation::sigmoid,
                          .derivative = activation::sigmoid_derivative};
    Criterion criterion{.function = criterion::mse, .derivative = criterion::mse_derivative};
    Network net({2, 4, 1}, activation, criterion);

    // Training dataset (XOR truth table)
    constexpr std::array training_data = {std::pair{std::array{0.0, 0.0}, std::array{0.0}},
                                          std::pair{std::array{0.0, 1.0}, std::array{1.0}},
                                          std::pair{std::array{1.0, 0.0}, std::array{1.0}},
                                          std::pair{std::array{1.0, 1.0}, std::array{0.0}}};

    // Training loop
    std::println("--- Training XOR network ---\n");

    constexpr int max_epochs = 750;
    for (int epoch{1}; epoch < max_epochs; ++epoch)
    {
        double epoch_loss = 0.0;

        for (const auto& [inputs, targets] : training_data)
        {
            net.feed_forward({inputs[0], inputs[1]});
            epoch_loss += net.compute_loss({targets[0]});
            net.back_propagate();
            net.step(0.3, 0.75);
        }

        epoch_loss /= training_data.size();
        std::println("Epoch {} | Loss: {:.4f}", epoch, epoch_loss);
    }

    // Test results
    std::println("\n--- Testing trained network ---\n");
    for (const auto& [inputs, targets] : training_data)
    {
        net.feed_forward({inputs[0], inputs[1]});
        std::println("[{}, {}] -> {:.4f} (target: {:.1f})", inputs[0], inputs[1],
                     net.get_output()[0], targets[0]);
    }

    return 0;
}
