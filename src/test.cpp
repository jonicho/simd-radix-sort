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

template <bool Combined, bool Up, typename BitSorter = BitSorterSequential,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
bool test(InputDistribution distribution, const std::size_t num,
          const unsigned int seed = time(NULL)) {
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
  std::cout << "Distribution: " << inputDistributionToString(distribution)
            << ", "
            << "Up: " << Up << ", " << BitSorter::name() << ", "
            << CmpSorter::name() << ", ";
  if constexpr (false && Combined && sizeof(DataElement<K, Ps...>) <= 8) {
    std::cout << "NOT TESTING" << std::endl;
    return true;
  }
  Data<K, Ps...> data(num, distribution, seed);
  const Data<K, Ps...> copyOfData(data);
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
        [&](auto *const... payloads) {
          sort<Up, BitSorter, CmpSorter>(16, data.num, data.keys, payloads...);
        },
        data.payloads);
  }
  // data.printBinary();
  // data.keys[1] = data.keys[0];
  // std::apply(
  //   [&](auto *const ...payloads) {
  //     ((payloads[1] = payloads[0]), ...);
  //   },
  //   data.payloads);
  const std::string errorMessage = data.checkData(Up, copyOfData);
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
bool testAllDistributions(const std::size_t num,
                          const unsigned int seed = time(NULL)) {
  if constexpr (Combined && !is_power_of_two<sizeof(DataElement<K, Ps...>)>) {
    return true;
  } else {
    bool passed = true;
    for (const auto distribution : std::vector<InputDistribution>{
             InputDistribution::Gaussian, InputDistribution::Uniform,
             InputDistribution::Zero, InputDistribution::ZeroOne,
             InputDistribution::Sorted, InputDistribution::ReverseSorted,
             InputDistribution::AlmostSorted,
             InputDistribution::AlmostReverseSorted}) {
      passed &= test<Combined, Up, BitSorter, CmpSorter, K, Ps...>(distribution,
                                                                   num, seed);
    }
    return passed;
  }
}

template <typename K, typename... Ps>
bool testSequential(const std::size_t num,
                    const unsigned int seed = time(NULL)) {
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
bool testSIMD(const std::size_t num, const unsigned int seed = time(NULL)) {
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
  const std::size_t maxNum = argc > 1 ? std::stoi(argv[1]) : 10000;
  const unsigned int seed = argc > 2 ? std::stoi(argv[2]) : time(NULL);
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
