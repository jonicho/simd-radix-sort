#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "cmp_sorters.hpp"
#include "common.hpp"
#include "data.hpp"
#include "radix_sort.hpp"

using namespace simd_sort;
using namespace simd_sort::radix_sort;

template <bool Combined, bool Up, InputDistribution Distribution,
          typename BitSorter = BitSorterSequential,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
bool test(std::size_t num, unsigned int seed = time(NULL)) {
  std::cout << "Testing: K: " << type_name<K> << ", ";
  if constexpr (sizeof...(Ps) > 0) {
    std::cout << "Ps: ";
    ((std::cout << type_name<Ps> << ", "), ...);
  }
  if constexpr (Combined) {
    std::cout << "Combined, ";
  } else {
    std::cout << "Separate, ";
  }
  std::cout << "Distribution: " << inputDistributionToString<Distribution>()
            << ", "
            << "Up: " << Up << ", " << BitSorter::name() << ", "
            << CmpSorter::name() << ", ";
  if constexpr (false && Combined && sizeof(DataElement<K, Ps...>) <= 8) {
    std::cout << "NOT TESTING" << std::endl;
    return true;
  }
  Data<K, Ps...> data(num, Distribution, seed);
  Data<K, Ps...> copyOfData(data);
  // std::cout << std::endl; data.printBinary();
  if constexpr (Combined) {
    DataElement<K, Ps...> *keysAndPayloads = (DataElement<K, Ps...> *)malloc(
        data.num * sizeof(DataElement<K, Ps...>));
    data.convertToSingleArray(keysAndPayloads);
    sort<Up, BitSorter, CmpSorter>(16, data.num, keysAndPayloads);
    data.setFromSingleArray(keysAndPayloads);
    free(keysAndPayloads);
  } else {
    std::apply(
        [&](auto *...payloads) {
          sort<Up, BitSorter, CmpSorter>(16, data.num, data.keys, payloads...);
        },
        data.payloads);
  }
  // data.printBinary();
  // data.keys[1] = data.keys[0];
  // std::apply(
  //   [&](auto *...payloads) {
  //     ((payloads[1] = payloads[0]), ...);
  //   },
  //   data.payloads);
  std::string errorMessage = data.checkData(Up, copyOfData);
  if (errorMessage.empty()) {
    std::cout << "passed" << std::endl;
    return true;
  } else {
    std::cout << "FAILED: " << errorMessage << std::endl;
    return false;
  }
}

template <bool Combined, bool Up, typename BitSorter = BitSorterSequential,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
bool testAllDistributions(std::size_t num, unsigned int seed = time(NULL)) {
  if constexpr (Combined && !is_power_of_two<sizeof(DataElement<K, Ps...>)>) {
    return true;
  } else {
    bool passed = true;
    passed &= test<Combined, Up, InputDistribution::Gaussian, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::Uniform, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::Zero, BitSorter, CmpSorter,
                   K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::ZeroOne, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::Sorted, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::ReverseSorted, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::AlmostSorted, BitSorter,
                   CmpSorter, K, Ps...>(num, seed);
    passed &= test<Combined, Up, InputDistribution::AlmostReverseSorted,
                   BitSorter, CmpSorter, K, Ps...>(num, seed);
    return passed;
  }
}

template <typename K, typename... Ps>
bool testSequential(std::size_t num, unsigned int seed = time(NULL)) {
  bool passed = true;
  passed &= testAllDistributions<false, true, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, false, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, true, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, false, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  return passed;
}

template <typename K, typename... Ps>
bool testSIMD(std::size_t num, unsigned int seed = time(NULL)) {
  bool passed = true;
  passed &= testAllDistributions<false, true, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, false, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, true, BitSorterSIMD<true>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, false, BitSorterSIMD<true>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, true, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, false, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, true, BitSorterSIMD<true>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, false, BitSorterSIMD<true>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);

  if constexpr ((sizeof...(Ps) == 0 ||
                 (sizeof...(Ps) == 1 &&
                  std::is_same_v<
                      K, std::tuple_element_t<0, std::tuple<Ps...>>>)) &&
                std::__is_one_of<K, int, double>::value) {
    passed &=
        testAllDistributions<true, BitSorterSIMD<false>,
                             CmpSorterBramasSmallSort, K, Ps...>(num, seed);
    passed &=
        testAllDistributions<true, BitSorterSIMD<true>,
                             CmpSorterBramasSmallSort, K, Ps...>(num, seed);
  }
  return passed;
}

int main(int argc, char const *argv[]) {
  std::size_t maxNum = 10000;
  if (argc > 1) {
    maxNum = std::stoi(argv[1]);
  }
  unsigned int seed = time(NULL);
  if (argc > 2) {
    seed = std::stoi(argv[2]);
  }
  bool passed = true;
  for (std::size_t num = 1; num <= maxNum; num *= 10) {
    std::cout << "Testing " << num << " elements" << std::endl;
    passed &= testSequential<uint8_t, uint32_t>(num, seed);
    passed &= testSequential<uint16_t, uint32_t>(num, seed);
    passed &= testSequential<uint32_t, uint32_t>(num, seed);
    passed &= testSequential<uint64_t, uint32_t>(num, seed);
    passed &= testSequential<int8_t, uint32_t>(num, seed);
    passed &= testSequential<int16_t, uint32_t>(num, seed);
    passed &= testSequential<int32_t, uint32_t>(num, seed);
    passed &= testSequential<int64_t, uint32_t>(num, seed);
    passed &= testSequential<float, uint32_t>(num, seed);
    passed &= testSequential<double, uint32_t>(num, seed);
    passed &= testSIMD<uint8_t, uint8_t>(num, seed);
    passed &= testSIMD<uint16_t, uint8_t>(num, seed);
    passed &= testSIMD<uint32_t, uint8_t>(num, seed);
    passed &= testSIMD<uint64_t, uint8_t>(num, seed);
    passed &= testSIMD<int8_t, uint8_t>(num, seed);
    passed &= testSIMD<int16_t, uint8_t>(num, seed);
    passed &= testSIMD<int32_t, uint8_t>(num, seed);
    passed &= testSIMD<int64_t, uint8_t>(num, seed);
    passed &= testSIMD<float, uint8_t>(num, seed);
    passed &= testSIMD<double, uint8_t>(num, seed);
    passed &= testSIMD<uint8_t, uint16_t>(num, seed);
    passed &= testSIMD<uint16_t, uint16_t>(num, seed);
    passed &= testSIMD<uint32_t, uint16_t>(num, seed);
    passed &= testSIMD<uint64_t, uint16_t>(num, seed);
    passed &= testSIMD<int8_t, uint16_t>(num, seed);
    passed &= testSIMD<int16_t, uint16_t>(num, seed);
    passed &= testSIMD<int32_t, uint16_t>(num, seed);
    passed &= testSIMD<int64_t, uint16_t>(num, seed);
    passed &= testSIMD<float, uint16_t>(num, seed);
    passed &= testSIMD<double, uint16_t>(num, seed);
    passed &= testSIMD<uint8_t, uint32_t>(num, seed);
    passed &= testSIMD<uint16_t, uint32_t>(num, seed);
    passed &= testSIMD<uint32_t, uint32_t>(num, seed);
    passed &= testSIMD<uint64_t, uint32_t>(num, seed);
    passed &= testSIMD<int8_t, uint32_t>(num, seed);
    passed &= testSIMD<int16_t, uint32_t>(num, seed);
    passed &= testSIMD<int32_t, uint32_t>(num, seed);
    passed &= testSIMD<int64_t, uint32_t>(num, seed);
    passed &= testSIMD<float, uint32_t>(num, seed);
    passed &= testSIMD<double, uint32_t>(num, seed);
    passed &= testSIMD<uint8_t, uint64_t>(num, seed);
    passed &= testSIMD<uint16_t, uint64_t>(num, seed);
    passed &= testSIMD<uint32_t, uint64_t>(num, seed);
    passed &= testSIMD<uint64_t, uint64_t>(num, seed);
    passed &= testSIMD<int8_t, uint64_t>(num, seed);
    passed &= testSIMD<int16_t, uint64_t>(num, seed);
    passed &= testSIMD<int32_t, uint64_t>(num, seed);
    passed &= testSIMD<int64_t, uint64_t>(num, seed);
    passed &= testSIMD<float, uint64_t>(num, seed);
    passed &= testSIMD<double, uint64_t>(num, seed);
    passed &= testSIMD<double, uint64_t, uint8_t>(num, seed);
  }

  if (passed) {
    std::cout << "All tests passed" << std::endl;
    return 0;
  } else {
    std::cout << "Tests failed, see above for details" << std::endl;
    return 1;
  }
}
