#include "network.hpp"
#include "activation.hpp"

#include <gtest/gtest.h>

using namespace ann;

class NetworkTest : public ::testing::Test
{
protected:
    Activation activation{.function = activation::tanh, .derivative = activation::tanh_derivative};
};

TEST_F(NetworkTest, CreationWithValidTopology)
{
    EXPECT_NO_THROW(Network({2, 4, 1}, activation));
}

TEST_F(NetworkTest, ThrowsOnEmptyTopology)
{
    EXPECT_THROW(Network({}, activation), std::invalid_argument);
}

TEST_F(NetworkTest, ThrowsOnTooFewInputs)
{
    Network net({2, 4, 1}, activation);

    EXPECT_THROW(net.feed_forward({0.0}), std::invalid_argument);
}

TEST_F(NetworkTest, ThrowsOnTooManyInputs)
{
    Network net({2, 4, 1}, activation);

    EXPECT_THROW(net.feed_forward({0.0, 0.0, 0.0}), std::invalid_argument);
}
