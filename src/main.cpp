#include "neuron.hpp"
#include "activation.hpp"

#include <print>

using namespace ann;

namespace
{

    auto create_input_neuron(double input, std::size_t index, std::size_t next_layer_size) -> Neuron
    {
        Neuron neuron(next_layer_size, index);
        neuron.set_output(input);

        return neuron;
    }

} // namespace

auto main() -> int
{
    // Input layer: 2 neurons
    std::vector<ann::Neuron> inputs;
    inputs.emplace_back(create_input_neuron(0.75, 0, 1));
    inputs.emplace_back(create_input_neuron(0.25, 1, 1));

    // Output layer: 1 neuron
    Neuron::Activation activation{.function = activation::linear,
                                  .derivative = activation::linear_derivative};
    Neuron output(0, 0, activation);

    // Forward pass
    output.feed_forward(inputs);
    std::println("Output: {:.4f}", output.get_output());

    return 0;
}
