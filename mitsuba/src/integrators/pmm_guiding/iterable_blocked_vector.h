/*
    This file is part of the implementation of the SIGGRAPH 2020 paper
    "Robust Fitting of Parallax-Aware Mixtures for Path Guiding",
    as well as the updated implementation of the ACM TOG 2019 paper
    "Volume Path Guiding Based on Zero-Variance Random Walk Theory".
    The implementation extends Mitsuba, a physically based rendering system.

    Copyright (c) 2020 Lukas Ruppert, Sebastian Herholz.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ITERABLEBLOCKEDVECTOR_H
#define ITERABLEBLOCKEDVECTOR_H

#include <mitsuba/mitsuba.h>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <numeric>

template<typename T>
class IterableBlockedVector
{
public:
    using block_bits = std::integral_constant<size_t, 16>;
    using block_size = std::integral_constant<size_t, 1<<block_bits::value>;

    using value_type             = T;
    using allocator_type         = std::allocator<T>;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using reference              = T&;
    using const_reference        = const T&;
    using pointer                = T*;
    using const_pointer          = const T*;

    class const_iterator;

    class iterator : public std::iterator<std::random_access_iterator_tag, value_type, difference_type, pointer, reference>
    {
    private:
        IterableBlockedVector* vector {nullptr};
        size_type pos {0};
    public:
        iterator() = default;

        explicit iterator(IterableBlockedVector* vector, size_type pos)
            : vector{vector}, pos{pos}
        {

        }
        iterator& operator++() {
            ++pos;
            return *this;
        }
        iterator operator++(int) {
            iterator copy{*this};
            ++pos;
            return copy;
        }
        iterator& operator--() {
            --pos;
            return *this;
        }
        iterator operator--(int) {
            iterator copy{*this};
            --pos;
            return copy;
        }
        bool operator<(iterator other) const {
            return vector < other.vector || (vector == other.vector && pos < other.pos);
        }
        bool operator>(iterator other) const {
            return vector > other.vector || (vector == other.vector && pos > other.pos);
        }
        bool operator<=(iterator other) const {
            return vector < other.vector || (vector == other.vector && pos <= other.pos);
        }
        bool operator>=(iterator other) const {
            return vector > other.vector || (vector == other.vector && pos >= other.pos);
        }
        bool operator==(iterator other) const {
            return vector == other.vector && pos == other.pos;
        }
        bool operator!=(iterator other) const {
            return vector != other.vector || pos != other.pos;
        }
        iterator& operator+=(difference_type offset) {
            pos += offset;
            return *this;
        }
        iterator operator+(difference_type offset) const {
            iterator copy{*this};
            copy += offset;
            return copy;
        }
        iterator& operator-=(difference_type offset) {
            pos -= offset;
            return *this;
        }
        iterator operator-(difference_type offset) const {
            iterator copy{*this};
            copy -= offset;
            return copy;
        }
        difference_type operator-(iterator other) const {
            return pos-other.pos;
        }
        reference operator[](difference_type offset) const {
            return *(*this+offset);
        }
        reference operator*() const {
            return (*vector->m_blocks[pos>>block_bits::value])[pos&(block_size::value-1)];
        }

        friend class const_iterator;
    };

    class const_iterator : public std::iterator<std::random_access_iterator_tag, value_type, difference_type, pointer, reference>
    {
    private:
        const IterableBlockedVector* vector {nullptr};
        size_type pos {0};
    public:
        const_iterator() = default;

        explicit const_iterator(const iterator& it)
            : vector{it.vector}, pos{it.pos}
        {

        }

        explicit const_iterator(const IterableBlockedVector* vector, size_type pos)
            : vector{vector}, pos{pos}
        {

        }
        const_iterator& operator++() {
            ++pos;
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator copy{*this};
            ++pos;
            return copy;
        }
        const_iterator& operator--() {
            --pos;
            return *this;
        }
        const_iterator operator--(int) {
            const_iterator copy{*this};
            --pos;
            return copy;
        }
        bool operator<(const_iterator other) const {
            return vector < other.vector || (vector == other.vector && pos < other.pos);
        }
        bool operator>(const_iterator other) const {
            return vector > other.vector || (vector == other.vector && pos > other.pos);
        }
        bool operator<=(const_iterator other) const {
            return vector < other.vector || (vector == other.vector && pos <= other.pos);
        }
        bool operator>=(const_iterator other) const {
            return vector > other.vector || (vector == other.vector && pos >= other.pos);
        }
        bool operator==(const_iterator other) const {
            return vector == other.vector && pos == other.pos;
        }
        bool operator!=(const_iterator other) const {
            return vector != other.vector || pos != other.pos;
        }
        const_iterator& operator+=(difference_type offset) {
            pos += offset;
            return *this;
        }
        const_iterator operator+(difference_type offset) const {
            const_iterator copy{*this};
            copy += offset;
            return copy;
        }
        const_iterator& operator-=(difference_type offset) {
            pos -= offset;
            return *this;
        }
        const_iterator operator-(difference_type offset) const {
            const_iterator copy{*this};
            copy -= offset;
            return copy;
        }
        difference_type operator-(const_iterator other) const {
            return pos-other.pos;
        }
        const_reference operator[](difference_type offset) const {
            return *(*this+offset);
        }
        const_reference operator*() const {
            return (*vector->m_blocks[pos>>block_bits::value])[pos&(block_size::value-1)];
        }
    };

    /*
     * reverse_iterator
     *
     * not implemented
     */

private:
    std::vector<std::unique_ptr<std::array<T, block_size::value>>> m_blocks;
    size_type m_size {0};

    void reserveBlocks(size_t numBlocks) {
        m_blocks.reserve(numBlocks);
        while (m_blocks.size() < numBlocks)
            m_blocks.emplace_back(new std::array<T, block_size::value>{});
    }

public:
    IterableBlockedVector() = default;

    ///merge contents
    explicit IterableBlockedVector(std::vector<IterableBlockedVector> &parts) {
        size_t numBlocks = 0;
        for (IterableBlockedVector& part : parts)
        {
            //drop empty blocks
            const size_t activeBlocksInPart = (part.m_size+block_size::value-1)>>block_bits::value;
            part.m_blocks.resize(activeBlocksInPart);
            numBlocks += activeBlocksInPart;
        }

        m_blocks.resize(numBlocks);

        size_t pos = 0;

        //first accumualte all full blocks
        for (IterableBlockedVector& part : parts)
        {
            if (part.empty())
                continue;

            size_type lastBlockSizePart = part.m_size&(block_size::value-1);

            for (auto blockIt = part.m_blocks.begin(); blockIt != part.m_blocks.end()-(lastBlockSizePart != 0); ++blockIt)
                m_blocks[pos++].swap(*blockIt);
        }

        m_size = block_size::value*pos;

        //then accumulate partial blocks and clear merged vectors
        for (IterableBlockedVector& part : parts)
        {
            size_type lastBlockSizePart = part.m_size&(block_size::value-1);

            //skip part consisting only out of full blocks, they have already been processed
            if (!lastBlockSizePart)
                continue;

            auto& lastBlockPart = part.m_blocks.back();
            auto& lastBlockMerged = m_blocks[pos];
            size_type lastBlockSizeMerged = m_size&(block_size::value-1);

            //swap the larger block into the merged vector
            if (lastBlockSizePart > lastBlockSizeMerged)
            {
                m_size += lastBlockSizePart-lastBlockSizeMerged;
                std::swap(lastBlockPart, lastBlockMerged);
                std::swap(lastBlockSizePart, lastBlockSizeMerged);
            }

            //continue with next part if an empty block has been swapped out
            if (!lastBlockSizePart)
                continue;

            //fill larger block using the smaller block
            const size_type freeSpace = block_size::value-lastBlockSizeMerged;
            const size_type copySize = std::min(freeSpace, lastBlockSizePart);

            std::copy(lastBlockPart->begin()+lastBlockSizePart-copySize, lastBlockPart->begin()+lastBlockSizePart, lastBlockMerged->begin()+lastBlockSizeMerged);
            m_size += copySize;
            lastBlockSizePart -= copySize;

            //continue if no data remains in the part's block
            if (!lastBlockSizePart)
                continue;

            //advance to next block and place the remainder there
            std::swap(m_blocks[++pos], lastBlockPart);
            m_size += lastBlockSizePart;
        }

        //increment pos for the partially filled block
        pos += (m_size&(block_size::value-1)) != 0;
        m_blocks.resize(pos);

        //cleanup
        for (IterableBlockedVector& part : parts)
            part.clear();
    }

    explicit IterableBlockedVector(const std::vector<T>& vector) {
        assign(vector.begin(), vector.end());
    }

    explicit IterableBlockedVector(size_type count) {
        resize(count);
    }

    void assign(size_type count, const T& value) {
        clear();
        reserve(count);
        auto blockIt = m_blocks.begin();
        while (count > block_size::value) {
            (*blockIt).fill(value);
            count -= block_size::value;
            ++blockIt;
        }
        if (count) {
            std::fill_n((*blockIt)->begin(), count, value);
        }
    }
    template<class InputIt>
    void assign(InputIt first, InputIt last) {
        clear();
        reserve(std::distance(first, last));
        auto blockIt = m_blocks.begin();
        while (std::distance(first, last) > static_cast<difference_type>(block_size::value)) {
            typename std::array<T, block_size::value>::iterator dest = (*blockIt)->begin();
            for (size_type i=0; i<block_size::value; ++i, ++dest, ++first)
                *dest = *first;
            ++blockIt;
        }
        if (first != last) {
            typename std::array<T, block_size::value>::iterator dest = (*blockIt)->begin();
            if (first != last) {
                *dest = *first;
                ++dest;
                ++first;
            }
        }
    }
    void assign(std::initializer_list<T> ilist) {
        assign(ilist.begin(), ilist.end());
    }

    //element access
    reference at(size_type pos) {
        if (pos >= m_size)
            throw std::out_of_range("IterableBlockedVector::at(): requested element is out of range.");
        return operator[](pos);
    }
    const_reference at(size_type pos) const {
        if (pos >= m_size)
            throw std::out_of_range("IterableBlockedVector::at(): requested element is out of range.");
        return operator[](pos);
    }

    reference operator[](size_type pos) {
        return (*m_blocks[pos>>block_bits::value])[pos&(block_size::value-1)];
    }
    const_reference operator[](size_type pos) const {
        return (*m_blocks[pos>>block_bits::value])[pos&(block_size::value-1)];
    }

    reference front() {
        return operator[](0);
    }
    const_reference front() const {
        return operator[](0);
    }

    reference back() {
        return operator[](m_size-1);
    }
    const_reference back() const {
        return operator[](m_size-1);
    }

    /*
     * data
     *
     * not implemented
     * - data does not reside in contiguous memory
     */

    //iterators
    iterator begin() noexcept {
        return iterator(this, 0);
    }
    const_iterator begin() const noexcept {
        return const_iterator(this, 0);
    }
    const_iterator cbegin() const noexcept {
        return const_iterator(this, 0);
    }

    iterator end() noexcept {
        return iterator(this, m_size);
    }
    const_iterator end() const noexcept {
        return const_iterator(this, m_size);
    }
    const_iterator cend() const noexcept {
        return const_iterator(this, m_size);
    }

    //reverse_iterator not implemented

    //capacity
    bool empty() const noexcept {
        return m_size == 0;
    }

    size_type size() const noexcept {
        return m_size;
    }

    size_type max_size() const noexcept {
        //make sure not to overflow the size_type
        return m_blocks.max_size()*block_size::value > m_blocks.max_size() ? m_blocks.size()*block_size::value : m_blocks.max_size();
    }

    void reserve(size_type new_cap) {
        reserveBlocks(new_cap>>block_bits::value);
    }

    size_type capacity() const noexcept {
        return m_blocks.size()*block_size::value;
    }

    void shrink_to_fit() {
        m_blocks.resize(m_size>>block_bits::value);
        m_blocks.shrink_to_fit();
    }

    //modifiers
    void clear() noexcept {
        m_blocks.clear();
        m_size = 0;
    }

    /*
     * insert
     * emplace
     * erase
     *
     * not implemented
     */

    void push_back(const T& value) {
        if ((m_size&(block_size::value-1)) == 0)
            reserveBlocks((m_size>>block_bits::value)+1);
        (*m_blocks[m_size>>block_bits::value])[m_size&(block_size::value-1)] = value;
        ++m_size;
    }
    void push_back(T&& value) {
        if ((m_size&(block_size::value-1)) == 0)
            reserveBlocks((m_size>>block_bits::value)+1);
        (*m_blocks[m_size>>block_bits::value])[m_size&(block_size::value-1)] = std::move(value);
        ++m_size;
    }

    template<class... Args>
    void emplace_back(Args&&... args) {
        if ((m_size&(block_size::value-1)) == 0)
            reserveBlocks((m_size>>block_bits::value)+1);
        T* element = &((*m_blocks[m_size>>block_bits::value])[m_size&(block_size::value-1)]);
        element->~T();
        new(element) T(std::forward<Args>(args)...);
        ++m_size;
    }

    void pop_back() {
        --m_size;
    }

    void resize(size_type count) {
        reserveBlocks((count+block_size::value-1)>>block_bits::value);
        m_size = count;
    }
    void resize(size_type count, const value_type& value) {
        reserveBlocks((count>>block_bits::value)+1);
        size_type pos = m_size;
        m_size = count;

        if (pos>>block_bits::value < m_size>>block_bits::value)
        {
            std::fill_n(&((*m_blocks[pos>>block_bits::value])[pos&(block_size::value-1)]), (block_size::value-1)-(pos&(block_size::value-1))+1, count);
            pos = (pos|(block_size::value-1))+1;
        }
        while (m_size-pos >= block_size::value)
        {
            m_blocks[pos>>block_bits::value]->fill(value);
            pos += block_size::value;
        }
        if (pos < m_size)
            std::fill_n(m_blocks[pos>>block_bits::value]->begin(), m_size-pos, value);
    }

    void swap(IterableBlockedVector& other) {
        m_blocks.swap(other.m_blocks);
        std::swap(m_size, other.m_size);
    }
};


#endif // ITERABLEBLOCKEDVECTOR_H
