//=============================================================================
//
// Generic AVX-512 implementation of MSB Radix Sort for sorting datasets
// with a key and an arbitrary number of payloads.
//
// By Jonas Richard Keller, 2023.
// This code is licensed under the MIT license.
//
// This code was written for the bachelor's thesis "SIMD Implementation of
// MSB Radix Sort with Separate Key and Payload Datastreams" at the
// Bielefeld University.
//
// The code was heavily inspired by the generic radix sort implementation
// by Prof. Dr.-Ing. Ralf Möller included in the T-SIMD library:
// https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html
//
//
// To sort a dataset, call the sort function like the following:
//
// simd_sort::radix_sort::sort(num, keyArray, payloadArrays...);
//
// where num is the number of elements in the dataset, keyArray is a pointer
// to the key array, and payloadArrays are the pointers to the payload arrays.
//
// Or, if key and payload datastreams are combined into a single array, call
// the sort function like the following:
//
// simd_sort::radix_sort::sort(num, (simd_sort::DataElement<K, Ps...> *)
// combinedArray);
//
// where num is the number of elements in the dataset, combinedArray is s
// pointer to the combined array and K and Ps... are the types of the key and
// the payloads.
//
// By default the sort function sorts in ascending order. To sort in descending
// order, call the sort function with the template parameter "Up" set to false:
// simd_sort::radix_sort::sort<false>(num, keyArray, payloadArrays...);
// or
// simd_sort::radix_sort::sort<false>(num, (simd_sort::DataElement<K, Ps...> *)
// combinedArray);
//
// For compilation with gcc, compile with the following flags:
// -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi2
//
// The library requires at least C++20.
//
//=============================================================================

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string>
#include <tuple>

#include "cmp_sorters.hpp"
#include "common.hpp"
#include "data.hpp"
#include "simd.hpp"

#if __has_include("../bramas/sort512.hpp")
#include "../bramas/sort512.hpp"
#endif

#if __has_include("../bramas/sort512kv.hpp")
#include "../bramas/sort512kv.hpp"
#endif

#ifdef __AVX512F__

namespace simd_sort::radix_sort {

template <typename T>
static inline bool isBitSet(const std::size_t bitNo, const T val) {
  return (std::bit_cast<UInt<sizeof(T)>>(val) >> bitNo) & 1;
}

template <typename K, typename... Ps>
static inline bool isBitSet(const std::size_t bitNo,
                            const DataElement<K, Ps...> val) {
  return isBitSet(bitNo, val.key);
}

namespace internal {
template <typename T>
struct KeyType {
  using type = T;
};
template <typename K, typename... Ps>
struct KeyType<DataElement<K, Ps...>> {
  using type = K;
};
}  // namespace internal

template <typename T>
using KeyType = typename internal::KeyType<T>::type;

template <typename T, bool Up, bool IsHighestBit, bool IsRightSide>
constexpr bool bitDirUp() {
  using K = KeyType<T>;
  if constexpr (std::is_integral_v<K>) {
    if constexpr (std::is_unsigned_v<K>) {
      return Up;
    } else {
      return IsHighestBit ? !Up : Up;
    }
  } else if constexpr (std::is_floating_point_v<K>) {
    return IsHighestBit ? !Up : IsRightSide;
  }
  return Up;
}

struct BitSorterSequential {
  static std::string name() { return "BitSorterSequential"; }

  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static inline SortIndex sortBit(std::size_t bitNo, SortIndex left,
                                  SortIndex right, K *keys, Ps *...payloads) {
    SortIndex l = left;
    SortIndex r = right;
    while (l <= r) {
      while (l <= r && (bitDirUp<K, Up, IsHighestBit, IsRightSide>() !=
                        isBitSet(bitNo, keys[l]))) {
        l++;
      }
      while (l <= r && (!bitDirUp<K, Up, IsHighestBit, IsRightSide>() !=
                        isBitSet(bitNo, keys[r]))) {
        r--;
      }
      if (l < r) {
        std::swap(keys[l], keys[r]);
        (std::swap(payloads[l], payloads[r]), ...);
      }
    }
    return l;
  }
};

struct BitSorterNoSort {
  static std::string name() { return "BitSorterNoSort"; }

  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static inline SortIndex sortBit(std::size_t bitNo, SortIndex left,
                                  SortIndex right, K *keys, Ps *...payloads) {
    return (left + right) / 2;
  }
};

template <bool OneReg = false>
struct BitSorterSIMD {
  static std::string name() {
    if constexpr (OneReg) {
      return "BitSorterSIMD<OneReg>";
    } else {
      return "BitSorterSIMD";
    }
  }

  template <typename K, typename... Ps>
  static constexpr SortIndex numElemsPerVec =
      OneReg ? 64 / std::max({sizeof(K), sizeof(Ps)...}) : 64 / sizeof(K);

  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static inline SortIndex sortBit(std::size_t bitNo, SortIndex left,
                                  SortIndex right, K *keys, Ps *...payloads) {
    static constexpr SortIndex _numElemsPerVec = numElemsPerVec<K, Ps...>;

    SortIndex numElems = right - left + 1;

    SortIndex readPosLeft = left;
    SortIndex readPosRight = right - _numElemsPerVec + 1;
    SortIndex writePosLeft = left;
    SortIndex writePosRight = right;

    simd::Vec<K, _numElemsPerVec * sizeof(K)> keyVecStore;
    std::tuple<simd::Vec<Ps, _numElemsPerVec * sizeof(Ps)>...> payloadVecStore;
    if (numElems >= _numElemsPerVec) {
      keyVecStore =
          simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosLeft]);
      payloadVecStore = std::make_tuple(
          simd::loadu<_numElemsPerVec * sizeof(Ps)>(&payloads[readPosLeft])...);
      readPosLeft += _numElemsPerVec;
    }

    while (readPosLeft <= readPosRight) {
      auto keyVec = keyVecStore;
      auto payloadVec = payloadVecStore;
      auto [sortMaskLeft, sortMaskRight] =
          getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVec, bitNo);
      SortIndex numElemsToLeft = simd::kpopcnt(sortMaskLeft);
      SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
      bool areEnoughElemsFreeLeft =
          (readPosLeft - writePosLeft) >= numElemsToLeft;
      if (areEnoughElemsFreeLeft) {
        keyVecStore =
            simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosRight]);
        payloadVecStore =
            std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
                &payloads[readPosRight])...);
        readPosRight -= _numElemsPerVec;
      } else {
        keyVecStore =
            simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosLeft]);
        payloadVecStore =
            std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
                &payloads[readPosLeft])...);
        readPosLeft += _numElemsPerVec;
      }
      compress_store_left_right(
          writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
          sortMaskRight, keyVec, payloadVec, keys, payloads...);
      writePosLeft += numElemsToLeft;
      writePosRight -= numElemsToRight;
    }

    SortIndex numElemsRest = readPosRight + _numElemsPerVec - readPosLeft;

    simd::Mask<_numElemsPerVec> restMask = 0;
    simd::Vec<K, _numElemsPerVec * sizeof(K)> keyVecRest;
    std::tuple<simd::Vec<Ps, _numElemsPerVec * sizeof(Ps)>...> payloadVecRest;
    if (numElemsRest != 0) {
      restMask = simd::kshiftr(simd::knot(simd::Mask<_numElemsPerVec>(0)),
                               _numElemsPerVec - numElemsRest);
      keyVecRest = simd::maskz_loadu<_numElemsPerVec * sizeof(K)>(
          restMask, &keys[readPosLeft]);
      payloadVecRest =
          std::make_tuple(simd::maskz_loadu<_numElemsPerVec * sizeof(Ps)>(
              restMask, &payloads[readPosLeft])...);
      readPosLeft += numElemsRest;
    }

    if (numElems >= _numElemsPerVec) {
      auto [sortMaskLeft, sortMaskRight] =
          getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVecStore,
                                                                bitNo);
      SortIndex numElemsToLeft = simd::kpopcnt(sortMaskLeft);
      SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
      compress_store_left_right(
          writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
          sortMaskRight, keyVecStore, payloadVecStore, keys, payloads...);
      writePosLeft += numElemsToLeft;
      writePosRight -= numElemsToRight;
    }

    if (numElemsRest != 0) {
      auto [sortMaskLeftRest, sortMaskRightRest] =
          getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVecRest,
                                                                bitNo);
      sortMaskLeftRest = simd::kand(sortMaskLeftRest, restMask);
      sortMaskRightRest = simd::kand(sortMaskRightRest, restMask);
      SortIndex numElemsToLeftRest = simd::kpopcnt(sortMaskLeftRest);
      SortIndex numElemsToRightRest = numElemsRest - numElemsToLeftRest;
      compress_store_left_right(writePosLeft, writePosLeft + numElemsToLeftRest,
                                sortMaskLeftRest, sortMaskRightRest, keyVecRest,
                                payloadVecRest, keys, payloads...);
      writePosLeft += numElemsToLeftRest;
      writePosRight -= numElemsToRightRest;
    }
    return writePosLeft;
  }

 private:
  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static inline std::tuple<simd::Mask<numElemsPerVec<K, Ps...>>,
                           simd::Mask<numElemsPerVec<K, Ps...>>>
  getSortMasks(simd::Vec<K, numElemsPerVec<K, Ps...> * sizeof(K)> keyVec,
               std::size_t bitNo) {
    if constexpr (bitDirUp<K, Up, IsHighestBit, IsRightSide>()) {
      auto sortMaskRight = simd::test_bit(keyVec, bitNo);
      auto sortMaskLeft = simd::knot(sortMaskRight);
      return std::make_tuple(sortMaskLeft, sortMaskRight);
    } else {
      auto sortMaskLeft = simd::test_bit(keyVec, bitNo);
      auto sortMaskRight = simd::knot(sortMaskLeft);
      return std::make_tuple(sortMaskLeft, sortMaskRight);
    }
  }

  template <typename K, typename... Ps>
  static inline void compress_store_left_right(
      SortIndex leftPos, SortIndex rightPos,
      simd::Mask<numElemsPerVec<K, Ps...>> leftMask,
      simd::Mask<numElemsPerVec<K, Ps...>> rightMask,
      simd::Vec<K, numElemsPerVec<K, Ps...> * sizeof(K)> keyVec,
      std::tuple<simd::Vec<Ps, numElemsPerVec<K, Ps...> * sizeof(Ps)>...>
          payloadVec,
      K *keys, Ps *...payloads) {
    simd::mask_compressstoreu(&keys[leftPos], leftMask, keyVec);
    std::apply(
        [&](auto... payloadVecs) {
          (simd::mask_compressstoreu(&payloads[leftPos], leftMask, payloadVecs),
           ...);
        },
        payloadVec);

    simd::mask_compressstoreu(&keys[rightPos], rightMask, keyVec);
    std::apply(
        [&](auto... payloadVecs) {
          (simd::mask_compressstoreu(&payloads[rightPos], rightMask,
                                     payloadVecs),
           ...);
        },
        payloadVec);
  }
};

template <bool Up, typename BitSorter, typename CmpSorter,
          bool IsRightSide = false, bool IsHighestBit = true, typename K,
          typename... Ps>
void radixRecursion(std::size_t bitNo, SortIndex cmpSortThreshold,
                    SortIndex left, SortIndex right, K *keys, Ps *...payloads) {
  if (right - left <= 0) {
    return;
  }
  if (right - left < cmpSortThreshold) {
    CmpSorter::template sort<Up, K, Ps...>(left, right, keys, payloads...);
    return;
  }

  SortIndex split =
      BitSorter::template sortBit<Up, IsHighestBit, IsRightSide, K, Ps...>(
          bitNo, left, right, keys, payloads...);
  if (bitNo > 0) {
    radixRecursion<Up, BitSorter, CmpSorter, IsHighestBit ? false : IsRightSide,
                   false>(bitNo - 1, cmpSortThreshold, left, split - 1, keys,
                          payloads...);
    radixRecursion<Up, BitSorter, CmpSorter, IsHighestBit ? true : IsRightSide,
                   false>(bitNo - 1, cmpSortThreshold, split, right, keys,
                          payloads...);
  }
}

template <bool Up = true, typename BitSorter = BitSorterSIMD<>,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
void sort(SortIndex cmpSortThreshold, SortIndex num, K *keys, Ps *...payloads) {
#if defined(SORT512_HPP) && defined(SORT512KV_HPP)
  if constexpr (std::is_same_v<CmpSorter, CmpSorterBramasSmallSort>) {
    // bramas small sort only supports as many elements as fit in 16 avx512
    // registers
    cmpSortThreshold =
        std::min(cmpSortThreshold, (SortIndex)(16 * 64 / sizeof(K)));
  }
#endif  // SORT512_HPP && SORT512KV_HPP
  radixRecursion<Up, BitSorter, CmpSorter>(sizeof(K) * 8 - 1, cmpSortThreshold,
                                           0, num - 1, keys, payloads...);
}

template <bool Up, typename BitSorter, typename CmpSorter, typename K,
          typename... Ps>
void sort(SortIndex cmpSortThreshold, SortIndex num,
          DataElement<K, Ps...> *elements) {
  static_assert(is_power_of_two<sizeof(DataElement<K, Ps...>)>,
                "size of DataElement<K, Ps...> must be a power of two");
#if defined(SORT512_HPP) && defined(SORT512KV_HPP)
  if constexpr (std::is_same_v<CmpSorter, CmpSorterBramasSmallSort>) {
    // bramas small sort only supports as many elements as fit in 16 avx512
    // registers
    cmpSortThreshold =
        std::min(cmpSortThreshold, (SortIndex)(16 * 64 / sizeof(K)));
  }
#endif  // SORT512_HPP && SORT512KV_HPP
  // static_assert(sizeof(K) >= (sizeof(Ps) + ...), "Sum of payload sizes must
  // be smaller than or equal to key size");
  radixRecursion<Up, BitSorter, CmpSorter>(sizeof(K) * 8 - 1, cmpSortThreshold,
                                           0, num - 1, elements);
}

template <bool Up = true, typename K, typename... Ps>
void sort(SortIndex num, K *keys, Ps *...payloads) {
  sort<Up, BitSorterSIMD<>, CmpSorterInsertionSort>(16, num, keys, payloads...);
}

}  // namespace simd_sort::radix_sort

#endif  // __AVX512F__
