#pragma once

#define MAX_K = 8 // todo: take care of SIMD vectors later

class GaussianMixtureModel {
public:
    uint32_t m_K {MAX_K}; // number of components
    std::array<Kernel, MAX_K> m_components;

    GaussianMixtureModel() = default;

    void initialize() {
        for (const auto& component : m_components) {
            component.initialize();
        }
    }
}