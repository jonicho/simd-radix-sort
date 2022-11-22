#include "perf.hpp"

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>

#include "lib/data.hpp"

#if __has_include("moeller/SIMDRadixSortGeneric.H")
#include "moeller/SIMDRadixSortGeneric.H"
#endif // __has_include("moeller/SIMDRadixSortGeneric.H")

#include "radixSort.hpp"

int main(int argc, char const *argv[]) {
  std::cout << std::fixed;
  std::cout << std::setprecision(6);
#ifdef _IPP_RADIX_IS_PRESENT_
  std::cout << "Initializing IPP..." << std::endl;
  ippRadix::ippInit();
#endif // _IPP_RADIX_IS_PRESENT_
  using AllSorts =
      PerfTest<SortMethodRadixSort<radixSort::BitSorterSequential,
                                   radixSort::CmpSorterInsertionSort>,
               SortMethodRadixSort<radixSort::BitSorterSIMD<true>,
                                   radixSort::CmpSorterInsertionSort>,
               SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                   radixSort::CmpSorterInsertionSort>,
               SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                   radixSort::CmpSorterBramasSmallSort>,
               SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                   radixSort::CmpSorterNoSort>,
#ifdef _SIMD_RADIX_SORT_GENERIC_H_
               SortMethodMoellerSeq, SortMethodMoellerCompress,
#endif // _SIMD_RADIX_SORT_GENERIC_H_
#ifdef _IPP_RADIX_IS_PRESENT_
               SortMethodIPPRadix,
#endif // _IPP_RADIX_IS_PRESENT_
               SortMethodSTLSort, SortMethodBramas, SortMethodBlacher>;
  using OnlyRadixSIMDNoCmp =
      PerfTest<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                   radixSort::CmpSorterNoSort>>;

  std::function<void()> testFunctions[] = {
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, float, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int8_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int16_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int32_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, float>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, float, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, double>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, double, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int8_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int16_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Uniform, int64_t>,

      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, float, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int8_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int16_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int32_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, float>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, float, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, double>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, double, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int8_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int16_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Gaussian, int64_t>,

      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, float, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int8_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int16_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int32_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, float>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, float, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, double>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, double, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int8_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int16_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::Zero, int64_t>,

      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, float, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int8_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int16_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int32_t, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, float>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, float, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, double>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, double, int64_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int8_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int16_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int32_t>,
      OnlyRadixSIMDNoCmp::thresh<InputDistribution::ZeroOne, int64_t>,

      AllSorts::thresh<InputDistribution::Uniform, float, int64_t>,
      AllSorts::thresh<InputDistribution::Uniform, int8_t, int64_t>,
      AllSorts::thresh<InputDistribution::Uniform, int16_t, int64_t>,
      AllSorts::thresh<InputDistribution::Uniform, int32_t, int64_t>,
      AllSorts::thresh<InputDistribution::Uniform, float>,
      AllSorts::thresh<InputDistribution::Uniform, float, int32_t>,
      AllSorts::thresh<InputDistribution::Uniform, double>,
      AllSorts::thresh<InputDistribution::Uniform, double, int64_t>,
      AllSorts::thresh<InputDistribution::Uniform, int8_t>,
      AllSorts::thresh<InputDistribution::Uniform, int16_t>,
      AllSorts::thresh<InputDistribution::Uniform, int32_t>,
      AllSorts::thresh<InputDistribution::Uniform, int64_t>,

#ifdef _SIMD_RADIX_SORT_GENERIC_H_
      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress,
                         InputDistribution::Gaussian>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress, InputDistribution::Uniform>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress, InputDistribution::Zero>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress, InputDistribution::ZeroOne>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress, InputDistribution::Sorted>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress,
                         InputDistribution::ReverseSorted>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress,
                         InputDistribution::AlmostSorted>,

      perfTestSpeedupAll<SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                             radixSort::CmpSorterInsertionSort>,
                         SortMethodMoellerCompress,
                         InputDistribution::AlmostReverseSorted>,
#endif // _SIMD_RADIX_SORT_GENERIC_H_

      perfTestSpeedupAllKP<
          SortMethodRadixSort<radixSort::BitSorterSIMD<true>,
                              radixSort::CmpSorterInsertionSort>,
          SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                              radixSort::CmpSorterInsertionSort>,
          InputDistribution::Uniform, false, 1>,

      AllSorts::perfTest<InputDistribution::Uniform, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::Gaussian, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::Zero, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::ZeroOne, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::Sorted, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, int32_t, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, int32_t,
                         int32_t>,

      AllSorts::perfTest<InputDistribution::Uniform, int32_t>,
      AllSorts::perfTest<InputDistribution::Gaussian, int32_t>,
      AllSorts::perfTest<InputDistribution::Zero, int32_t>,
      AllSorts::perfTest<InputDistribution::ZeroOne, int32_t>,
      AllSorts::perfTest<InputDistribution::Sorted, int32_t>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, int32_t>,

      AllSorts::perfTest<InputDistribution::Uniform, float>,
      AllSorts::perfTest<InputDistribution::Gaussian, float>,
      AllSorts::perfTest<InputDistribution::Zero, float>,
      AllSorts::perfTest<InputDistribution::ZeroOne, float>,
      AllSorts::perfTest<InputDistribution::Sorted, float>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, float>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, float>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, float>,

      AllSorts::perfTest<InputDistribution::Uniform, float, int32_t>,
      AllSorts::perfTest<InputDistribution::Gaussian, float, int32_t>,
      AllSorts::perfTest<InputDistribution::Zero, float, int32_t>,
      AllSorts::perfTest<InputDistribution::ZeroOne, float, int32_t>,
      AllSorts::perfTest<InputDistribution::Sorted, float, int32_t>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, float, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, float, int32_t>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, float,
                         int32_t>,

      AllSorts::perfTest<InputDistribution::Uniform, double>,
      AllSorts::perfTest<InputDistribution::Gaussian, double>,
      AllSorts::perfTest<InputDistribution::Zero, double>,
      AllSorts::perfTest<InputDistribution::ZeroOne, double>,
      AllSorts::perfTest<InputDistribution::Sorted, double>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, double>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, double>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, double>,

      AllSorts::perfTest<InputDistribution::Uniform, uint8_t>,
      AllSorts::perfTest<InputDistribution::Gaussian, uint8_t>,
      AllSorts::perfTest<InputDistribution::Zero, uint8_t>,
      AllSorts::perfTest<InputDistribution::ZeroOne, uint8_t>,
      AllSorts::perfTest<InputDistribution::Sorted, uint8_t>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, uint8_t>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, uint8_t>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, uint8_t>,

      AllSorts::perfTest<InputDistribution::Uniform, int16_t>,
      AllSorts::perfTest<InputDistribution::Gaussian, int16_t>,
      AllSorts::perfTest<InputDistribution::Zero, int16_t>,
      AllSorts::perfTest<InputDistribution::ZeroOne, int16_t>,
      AllSorts::perfTest<InputDistribution::Sorted, int16_t>,
      AllSorts::perfTest<InputDistribution::ReverseSorted, int16_t>,
      AllSorts::perfTest<InputDistribution::AlmostSorted, int16_t>,
      AllSorts::perfTest<InputDistribution::AlmostReverseSorted, int16_t>,

      AllSorts::perfTestNum<InputDistribution::Uniform, int32_t>,
      AllSorts::perfTestNum<InputDistribution::Zero, int32_t>,
  };

  for (auto &testFunction : testFunctions) {
    testFunction();
  }
}
