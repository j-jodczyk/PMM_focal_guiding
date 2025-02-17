#include "gtest/gtest.h"

#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_mixture_model.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/2d_env.h"

class GMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        numComponents = 3;
        maxNumComponents = 5;
        minNumComponents = 2;

        gmm.setMaxNumComp(maxNumComponents);
        gmm.setMinNumComp(minNumComponents);

        Env2D::AABB aabb = {{ -10, -10 }, { 10, 10 }};
        gmm.init(numComponents, 2, aabb);
    }

    pmm_focal::GaussianMixtureModel<double, Env2D> gmm;
    size_t numComponents;
    size_t maxNumComponents;
    size_t minNumComponents;
};

TEST_F(GMMTest, InitializationTest) {
    EXPECT_EQ(gmm.getMinNumComp(), minNumComponents);
    EXPECT_EQ(gmm.getMaxNumComp(), maxNumComponents);
    EXPECT_EQ(gmm.getNumActiveComponents(), numComponents);
}

TEST_F(GMMTest, SplitComponentTest) {
    gmm.splitComponent(0);
    EXPECT_EQ(gmm.getNumActiveComponents(), numComponents + 1);
    std::vector<pmm_focal::GaussianComponent> components = gmm.getComponents();
    EXPECT_EQ(components[4].getWeight(), 0);
    for (int i=0; i<4; ++i) {
        EXPECT_GT(components[i].getWeight(), 0) << "Weight should be greater than 0";
    }
}

TEST_F(GMMTest, MergeComponentsTest) {
    gmm.splitComponent(0);
    gmm.mergeComponents(0, 1);
    EXPECT_EQ(gmm.getNumActiveComponents(), numComponents);
    std::vector<pmm_focal::GaussianComponent> components = gmm.getComponents();
    EXPECT_EQ(components[3].getWeight(), 0);
    EXPECT_EQ(components[4].getWeight(), 0);
    for (int i=0; i<3; ++i) {
        EXPECT_GT(components[i].getWeight(), 0) << "Weight should be greater than 0";
    }
}

TEST_F(GMMTest, DeactivateComponentTest) {
    gmm.deactivateComponent(0);
    std::vector<pmm_focal::GaussianComponent> components = gmm.getComponents();
    EXPECT_GT(components[0].getWeight(), 0) << "Weight should be greater than 0";
    EXPECT_GT(components[1].getWeight(), 0) << "Weight should be greater than 0";
    for (int i=2; i<5; ++i) {
        EXPECT_EQ(components[i].getWeight(), 0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
