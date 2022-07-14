#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include "bramas/sort512.hpp"
#include "bramas/sort512kv.hpp"

#include "lib/data.hpp"
#include "lib/type_name.hpp"

#include "radixSort.hpp"

using namespace radixSort;

template <bool Up, InputDistribution Distribution,
          typename BitSorter = BitSorterSequential,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
bool test(std::size_t num, uint seed = time(NULL)) {
  std::cout << "Testing: K: " << type_name<K> << ", ";
  if constexpr (sizeof...(Ps) > 0) {
    std::cout << "Ps: ";
    ((std::cout << type_name<Ps> << ", "), ...);
  }
  std::cout << "Distribution: " << inputDistributionToString<Distribution>()
            << ", "
            << "Up: " << Up << ", " << BitSorter::name() << ", "
            << CmpSorter::name() << ", ";
  InputData<K, Ps...> data(num, Distribution, seed);
  InputData<K, Ps...> copyOfData(data);
  // data.printBinary();
  std::apply(
      [&](auto *...payloads) {
        sort<Up, BitSorter, CmpSorter>(16, data.num, data.keys, payloads...);
      },
      data.payloads);
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

template <bool Up, typename BitSorter = BitSorterSequential,
          typename CmpSorter = CmpSorterInsertionSort, typename K,
          typename... Ps>
bool testAllDistributions(std::size_t num, uint seed = time(NULL)) {
  bool passed = true;
  passed &=
      test<Up, InputDistribution::Gaussian, BitSorter, CmpSorter, K, Ps...>(
          num, seed);
  passed &=
      test<Up, InputDistribution::Uniform, BitSorter, CmpSorter, K, Ps...>(
          num, seed);
  passed &= test<Up, InputDistribution::Zero, BitSorter, CmpSorter, K, Ps...>(
      num, seed);
  passed &=
      test<Up, InputDistribution::ZeroOne, BitSorter, CmpSorter, K, Ps...>(
          num, seed);
  passed &= test<Up, InputDistribution::Sorted, BitSorter, CmpSorter, K, Ps...>(
      num, seed);
  passed &= test<Up, InputDistribution::ReverseSorted, BitSorter, CmpSorter, K,
                 Ps...>(num, seed);
  passed &=
      test<Up, InputDistribution::AlmostSorted, BitSorter, CmpSorter, K, Ps...>(
          num, seed);
  passed &= test<Up, InputDistribution::AlmostReverseSorted, BitSorter,
                 CmpSorter, K, Ps...>(num, seed);

  return passed;
}

template <typename K, typename... Ps>
bool testSequential(std::size_t num, uint seed = time(NULL)) {
  bool passed = true;
  passed &= testAllDistributions<true, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, BitSorterSequential,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  return passed;
}

template <typename K, typename... Ps>
bool testSIMD(std::size_t num, uint seed = time(NULL)) {
  bool passed = true;
  passed &= testAllDistributions<true, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, BitSorterSIMD<false>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<true, BitSorterSIMD<true>,
                                 CmpSorterInsertionSort, K, Ps...>(num, seed);
  passed &= testAllDistributions<false, BitSorterSIMD<true>,
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
  uint seed = time(NULL);
  if (argc > 2) {
    seed = std::stoi(argv[2]);
  }
  bool passed = true;
#if 1
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
#endif

  if (passed) {
    std::cout << "All tests passed" << std::endl;
    return 0;
  } else {
    std::cout << "Tests failed, see above for details" << std::endl;
    return 1;
  }
}
