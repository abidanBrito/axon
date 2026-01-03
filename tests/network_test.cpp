#include "network.hpp"
#include "activation.hpp"
#include "criterion.hpp"

#include <gtest/gtest.h>

using namespace ann;

class NetworkTest : public ::testing::Test
{
protected:
    Activation activation{.function = activation::tanh, .derivative = activation::tanh_derivative};
    Criterion criterion{.function = criterion::mse, .derivative = criterion::mse_derivative};
};

TEST_F(NetworkTest, CreationWithValidTopology)
{
    EXPECT_NO_THROW(Network({2, 4, 1}, activation, criterion));
}

TEST_F(NetworkTest, ThrowsOnEmptyTopology)
{
    EXPECT_THROW(Network({}, activation, criterion), std::invalid_argument);
}

TEST_F(NetworkTest, ThrowsOnTooFewInputs)
{
    Network net({2, 4, 1}, activation, criterion);

    EXPECT_THROW(net.feed_forward({0.0}), std::invalid_argument);
}

TEST_F(NetworkTest, ThrowsOnTooManyInputs)
{
    Network net({2, 4, 1}, activation, criterion);

    EXPECT_THROW(net.feed_forward({0.0, 0.0, 0.0}), std::invalid_argument);
}

TEST_F(NetworkTest, FeedForwardProducesOutput)
{
    Network net({2, 3, 1}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    auto output = net.get_output();

    EXPECT_EQ(output.size(), 1);
    EXPECT_GE(output[0], -1.0);
    EXPECT_LE(output[0], 1.0);
}

TEST_F(NetworkTest, ComputeLossReturnsNonNegativeValue)
{
    Network net({2, 3, 1}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    double loss = net.compute_loss({0.0});

    EXPECT_GE(loss, 0.0);
}

TEST_F(NetworkTest, LossDecreasesWithPerfectPrediction)
{
    Network net({2, 3, 1}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    auto output = net.get_output();

    double loss = net.compute_loss({output[0]});

    EXPECT_NEAR(loss, 0.0, 1e-10);
}

TEST_F(NetworkTest, BackPropagateDoesNotThrow)
{
    Network net({2, 3, 1}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    net.compute_loss({0.0});

    EXPECT_NO_THROW(net.back_propagate());
}

TEST_F(NetworkTest, StepDoesNotThrow)
{
    Network net({2, 3, 1}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    net.compute_loss({0.0});
    net.back_propagate();

    EXPECT_NO_THROW(net.step(0.1, 0.5));
}

TEST_F(NetworkTest, MultipleOutputsWork)
{
    Network net({2, 4, 3}, activation, criterion);

    net.feed_forward({0.5, 0.5});
    auto output = net.get_output();

    EXPECT_EQ(output.size(), 3);
}
