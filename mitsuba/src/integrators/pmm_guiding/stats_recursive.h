#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

#include <string>
#include <vector>

namespace pmm_focal {

using Spectrum = mitsuba::Spectrum;
using ImageBlock = mitsuba::ImageBlock;

template<template<typename Data> class Entry>
struct StatsRecursive {
    using Float = mitsuba::Float;

    template<typename T, int Count>
    struct Stack {
        Stack(const std::string &name) {}
        void reset() {}
        void add(int depth, const T &value, Float weight = 1) {}
    };

    Entry<Spectrum>    albedo          { "pixel.albedo"    };
    Entry<Float>       roughness       { "pixel.roughness" };
    Entry<Float>       guidingPdf      { "d.pdf"           };
    Entry<Float>       avgPathLength   { "paths.length"    };
    Entry<Float>       numPaths        { "paths.count"     };
    Stack<Spectrum, 6> emitted         { "e.emitted"       };

    void reset() {
        albedo.reset();
        roughness.reset();
        guidingPdf.reset();
        avgPathLength.reset();
        numPaths.reset();
        emitted.reset();
    }
};

template<typename T>
struct FormatDescriptor {};

template<>
struct FormatDescriptor<mitsuba::Float> {
    int numComponents = 1;
    mitsuba::Bitmap::EPixelFormat pixelFormat = mitsuba::Bitmap::EPixelFormat::ELuminance;
    std::string pixelName = "luminance";
};

template<>
struct FormatDescriptor<mitsuba::Spectrum> {
    int numComponents = SPECTRUM_SAMPLES;
    mitsuba::Bitmap::EPixelFormat pixelFormat = mitsuba::Bitmap::EPixelFormat::ESpectrum;
    std::string pixelName = "rgb";
};

struct StatsRecursiveImageBlockCache {
    thread_local static StatsRecursiveImageBlockCache *instance;
    StatsRecursiveImageBlockCache(std::function<ImageBlock *()> createImage)
    : createImage(createImage) {
        instance = this;
    }

    std::function<mitsuba::ImageBlock *()> createImage;
    mutable std::vector<mitsuba::ref<mitsuba::ImageBlock>> blocks;
};

template<typename T>
struct StatsRecursiveImageBlockEntry {
    StatsRecursiveImageBlockEntry(const std::string &) {
        image = StatsRecursiveImageBlockCache::instance->createImage();
        image->setWarn(false); // some statistics can be negative
        StatsRecursiveImageBlockCache::instance->blocks.push_back(image);
    }

    ImageBlock *image;

    void add(const T &, mitsuba::Float) {}
};

struct StatsRecursiveImageBlocks : StatsRecursiveImageBlockCache, StatsRecursive<StatsRecursiveImageBlockEntry> {
    StatsRecursiveImageBlocks(std::function<ImageBlock *()> createImage)
    : StatsRecursiveImageBlockCache(createImage) {}

    void clear() {
        for (auto &block : blocks)
            block->clear();
    }

    void put(StatsRecursiveImageBlocks &other) const {
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i]->put(other.blocks[i]);
        }
    }

    std::vector<mitsuba::Bitmap *> getBitmaps() {
        std::vector<mitsuba::Bitmap *> result;
        for (auto &block : blocks)
            result.push_back(block->getBitmap());
        return result;
    }
};

struct StatsRecursiveDescriptorCache {
    thread_local static StatsRecursiveDescriptorCache *instance;
    StatsRecursiveDescriptorCache() {
        instance = this;
    }

    std::string names = "color", types = "rgb";

    int size = 1;
    int components = SPECTRUM_SAMPLES;
};

template<typename T>
struct StatsRecursiveDescriptorEntry {
    StatsRecursiveDescriptorEntry(const std::string &name) {
        auto &cache = *StatsRecursiveDescriptorCache::instance;

        cache.names += ", " + name;

        FormatDescriptor<T> fmt;
        cache.components += fmt.numComponents;
        cache.types += ", " + fmt.pixelName;

        cache.size += 1;
    }

    void add(const T &, mitsuba::Float) {}
};

struct StatsRecursiveDescriptor : StatsRecursiveDescriptorCache, StatsRecursive<StatsRecursiveDescriptorEntry> {
};

struct StatsRecursiveValuesCache {
    thread_local static StatsRecursiveValuesCache *instance;
    StatsRecursiveValuesCache() {
        instance = this;
    }

    std::vector<std::function<void (ImageBlock *, const mitsuba::Point2 &, mitsuba::Float)>> putters;
};

template<typename T>
struct StatsRecursiveValueEntry {
    StatsRecursiveValueEntry(const std::string &) {
        StatsRecursiveValuesCache::instance->putters.push_back([&](ImageBlock *block, const mitsuba::Point2 &samplePos, mitsuba::Float alpha) {
            Spectrum spec { value };
            mitsuba::Float temp[SPECTRUM_SAMPLES + 2];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                temp[i] = spec[i];
            temp[SPECTRUM_SAMPLES] = 1.0f;
            temp[SPECTRUM_SAMPLES + 1] = weight > 0 ? weight : 1;
            block->put(samplePos, temp);
        });
    }

    T value { 0.f };
    mitsuba::Float weight = 0.f;

    void reset() {
        value = T { 0.f };
        weight = 0.f;
    }

    void increment() {
        value++;
    }

    void add(const T &v, mitsuba::Float w = 1) {
        value += v;
        weight += w;
    }
};

struct StatsRecursiveValues : StatsRecursiveValuesCache, StatsRecursive<StatsRecursiveValueEntry> {
    void put(StatsRecursiveImageBlocks &other, const mitsuba::Point2 &samplePos, Float alpha) {
        for (size_t i = 0; i < putters.size(); ++i)
            putters[i](other.blocks[i], samplePos, alpha);
    }
};

};
