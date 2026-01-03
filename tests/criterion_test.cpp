#include "criterion.hpp"

#include <gtest/gtest.h>

using namespace ann::criterion;

TEST(CriterionTest, MSEIsSymmetric)
{
    double error1 = mse(1.0, 0.5);
    double error2 = mse(0.5, 1.0);

    EXPECT_DOUBLE_EQ(error1, error2);
}

TEST(CriterionTest, MSEIsAlwaysNonNegative)
{
    EXPECT_GE(mse(1.0, 0.0), 0.0);
    EXPECT_GE(mse(0.0, 1.0), 0.0);
    EXPECT_GE(mse(-1.0, 1.0), 0.0);
}

TEST(CriterionTest, MSEValueIsCorrect)
{
    EXPECT_DOUBLE_EQ(mse(1.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(mse(0.5, 0.0), 0.25);
    EXPECT_DOUBLE_EQ(mse(1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(mse(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(mse(-1.0, -1.0), 0.0);
}

TEST(CriterionTest, MSEDerivativeSignIsCorrect)
{
    EXPECT_GT(mse_derivative(0.0, 1.0), 0.0);
    EXPECT_LT(mse_derivative(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(mse_derivative(0.5, 0.5), 0.0);
}

TEST(CriterionTest, MSEDerivativeValueIsCorrect)
{
    EXPECT_DOUBLE_EQ(mse_derivative(1.0, 0.5), -0.5);
    EXPECT_DOUBLE_EQ(mse_derivative(0.0, 1.0), 1.0);
}
