#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <time.h>
#include <type_traits>
#include <x86intrin.h>

#include "blacher/avx2sort.h"
#include "bramas/sort512.hpp"
#include "bramas/sort512kv.hpp"
#include "lib/data.hpp"
#include "lib/ippRadix.hpp"
#include "lib/type_name.hpp"

#if __has_include("moeller/SIMDRadixSortGeneric.H")
#include "moeller/SIMDRadixSortGeneric.H"
#endif

#include "radixSort.hpp"

const std::string dataDir = "/tmp/radixSortData";

template <int Bytes> struct _UInt;
template <> struct _UInt<1> { using type = uint8_t; };
template <> struct _UInt<2> { using type = uint16_t; };
template <> struct _UInt<4> { using type = uint32_t; };
template <> struct _UInt<8> { using type = uint64_t; };
#ifdef _SIMD_RADIX_SORT_GENERIC_H_
template <> struct _UInt<16> { using type = radix::uint128_t; };
#endif

template <int Bytes> using UInt = typename _UInt<Bytes>::type;

template <typename BitSorter, typename CmpSorter> struct SortMethodRadixSort {
  static std::string name() {
    std::string result = "Radix";
    if constexpr (std::is_same_v<BitSorter, radixSort::BitSorterSequential>) {
      result += "Seq";
    } else if constexpr (std::is_same_v<BitSorter,
                                        radixSort::BitSorterSIMD<false>>) {
      result += "SIMD";
    } else if constexpr (std::is_same_v<BitSorter,
                                        radixSort::BitSorterSIMD<true>>) {
      result += "SIMDOneReg";
    }
    if constexpr (std::is_same_v<CmpSorter,
                                 radixSort::CmpSorterBramasSmallSort>) {
      result += "BramSmall";
    } else if constexpr (std::is_same_v<CmpSorter,
                                        radixSort::CmpSorterNoSort>) {
      result += "NoCmp";
    }
    return result;
  }
  static constexpr bool areKeyAndPayloadSeparate = true;
  static constexpr bool hasThreshold = true;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    // if the one reg version does not have a difference, don't test it
    if constexpr (std::is_same_v<BitSorter, radixSort::BitSorterSIMD<true>>) {
      if constexpr (std::max({std::size_t(0), sizeof(Ps)...}) <= sizeof(K)) {
        return false;
      }
    }
    if constexpr (std::is_same_v<CmpSorter,
                                 radixSort::CmpSorterBramasSmallSort>) {
      if constexpr (sizeof...(Ps) == 0) {
        return std::is_same_v<K, int> || std::is_same_v<K, double>;
      }
      if constexpr (sizeof...(Ps) == 1) {
        return std::is_same_v<K, int> && std::is_same_v<int, Ps...>;
      }
      return false;
    }
    return true;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num, K *keys, Ps *...payloads) {
    if constexpr (std::is_same_v<CmpSorter,
                                 radixSort::CmpSorterBramasSmallSort>) {
      sortThresh(16 * 64 / sizeof(K), num, keys, payloads...);
    } else {
      if constexpr (std::is_same_v<BitSorter, radixSort::BitSorterSequential>) {
        sortThresh(64, num, keys, payloads...);
      } else {
        sortThresh(16, num, keys, payloads...);
      }
    }
  }

  template <typename K, typename... Ps>
  static void sortThresh(radixSort::SortIndex cmpSortThresh,
                         radixSort::SortIndex num, K *keys, Ps *...payloads) {
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    radixSort::sort<true, BitSorter, CmpSorter>(cmpSortThresh, num, keys,
                                                payloads...);
  }
};

#ifdef _SIMD_RADIX_SORT_GENERIC_H_
struct SortMethodMoellerCompress {
  static std::string name() { return "MoellerCompress"; }
  static constexpr bool areKeyAndPayloadSeparate = false;
  static constexpr bool hasThreshold = true;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    if constexpr (sizeof...(Ps) > 1) {
      return false;
    }
    if constexpr (!((sizeof(K) >= sizeof(Ps)) && ...)) {
      return false;
    }
    return true;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num,
                   InputDataElement<K, Ps...> *elems) {
    sortThresh(16, num, elems);
  }

  template <typename K, typename... Ps>
  static void sortThresh(radix::SortIndex cmpSortThresh,
                         radixSort::SortIndex num,
                         InputDataElement<K, Ps...> *elems) {
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    radix::simdRadixSortCompress<K, true,
                                 UInt<sizeof(InputDataElement<K, Ps...>)>>(
        (UInt<sizeof(InputDataElement<K, Ps...>)> *)elems, 0, num - 1,
        cmpSortThresh);
  }
};

struct SortMethodMoellerSeq {
  static std::string name() { return "MoellerSeq"; }
  static constexpr bool areKeyAndPayloadSeparate = false;
  static constexpr bool hasThreshold = true;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    if constexpr (sizeof...(Ps) > 1) {
      return false;
    }
    if constexpr (!((sizeof(K) == sizeof(Ps)) && ...)) {
      return false;
    }
    return true;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num,
                   InputDataElement<K, Ps...> *elems) {
    sortThresh(64, num, elems);
  }

  template <typename K, typename... Ps>
  static void sortThresh(radix::SortIndex cmpSortThresh,
                         radixSort::SortIndex num,
                         InputDataElement<K, Ps...> *elems) {
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    radix::seqRadixSort<K, true, UInt<sizeof(InputDataElement<K, Ps...>)>>(
        (UInt<sizeof(InputDataElement<K, Ps...>)> *)elems, 0, num - 1,
        cmpSortThresh);
  }
};
#endif

struct SortMethodSTLSort {
  static std::string name() { return "STLSort"; }
  static constexpr bool areKeyAndPayloadSeparate = false;
  static constexpr bool hasThreshold = false;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    return true;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num,
                   InputDataElement<K, Ps...> *keysAndPayloads) {
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    std::sort(keysAndPayloads, keysAndPayloads + num);
  }
};

#ifdef _IPP_RADIX_IS_PRESENT_
struct SortMethodIPPRadix {
  static std::string name() { return "IPPRadix"; }
  static constexpr bool areKeyAndPayloadSeparate = false;
  static constexpr bool hasThreshold = false;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    return sizeof...(Ps) == 0 && !std::is_same_v<K, int8_t>;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num,
                   InputDataElement<K, Ps...> *keysAndPayloads) {
    static_assert(sizeof...(Ps) == 0, "IPPRadix does not support payloads");
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    ippRadix::sort((K *)keysAndPayloads, num);
  }
};
#endif

struct SortMethodBramas {
  static std::string name() { return "BramasSort"; }
  static constexpr bool areKeyAndPayloadSeparate = true;
  static constexpr bool hasThreshold = false;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    if constexpr (sizeof...(Ps) == 0) {
      return std::is_same_v<K, int> || std::is_same_v<K, double>;
    }
    if constexpr (sizeof...(Ps) == 1) {
      return std::is_same_v<K, int> && std::is_same_v<int, Ps...>;
    }
    return false;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num, K *keys, Ps *...payloads) {
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    static_assert(sizeof...(Ps) == 0 || sizeof...(Ps) == 1,
                  "SortMethodBramas does not support more than 1 payload");
    if constexpr (sizeof...(Ps) == 1) {
      static_assert(
          std::is_same_v<K, Ps...>,
          "SortMethodBramas requires the same type for key and payload");
    }
    if constexpr (sizeof...(Ps) == 0) {
      Sort512::Sort(keys, num);
    } else if constexpr (sizeof...(Ps) == 1) {
      Sort512kv::Sort(keys, payloads..., num);
    } else {
    }
  }
};

struct SortMethodBlacher {
  static std::string name() { return "BlacherSort"; }
  static constexpr bool areKeyAndPayloadSeparate = true;
  static constexpr bool hasThreshold = false;

  template <typename K, typename... Ps> static constexpr bool isSupported() {
    return sizeof...(Ps) == 0 && std::is_same_v<K, int32_t>;
  }

  template <typename K, typename... Ps>
  static void sort(radixSort::SortIndex num, K *keys, Ps *...payloads) {
    static_assert(sizeof...(Ps) == 0 && std::is_same_v<K, int32_t>,
                  "SortMethodBlacher only supports int32_t key and no payload");
    static_assert(isSupported<K, Ps...>(), "Unsupported type combination");
    avx2::quicksort(keys, num);
  }
};

template <typename SortMethod, typename K, typename... Ps>
double measureTimePerElement(const InputData<K, Ps...> &data) {
  InputData<K, Ps...> copyOfData(data);
  struct timespec start, end;
  if constexpr (SortMethod::areKeyAndPayloadSeparate) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    std::apply(
        [&](auto *...payloads) {
          SortMethod::template sort(copyOfData.num, copyOfData.keys,
                                    payloads...);
        },
        copyOfData.payloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
  } else {
    InputDataElement<K, Ps...> *keysAndPayloads =
        (InputDataElement<K, Ps...> *)malloc(
            copyOfData.num * sizeof(InputDataElement<K, Ps...>));
    copyOfData.convertToSingleArray(keysAndPayloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    SortMethod::template sort<K, Ps...>(copyOfData.num, keysAndPayloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    copyOfData.setFromSingleArray(keysAndPayloads);
    free(keysAndPayloads);
  }
  if (SortMethod::name().find("NoCmp") == std::string::npos &&
      !copyOfData.isSorted(true)) {
    std::cout << "ERROR: "
              << "not sorted (" + std::to_string(copyOfData.numUnsorted(true)) +
                     " out of " + std::to_string(copyOfData.num - 1) + " pairs)"
              << "(" << data.num << " " << SortMethod::name() << ")"
              << std::endl;
    exit(-1);
  }
  double nanoSeconds =
      (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
  return nanoSeconds / (double)data.num;
}

template <typename SortMethod, typename K, typename... Ps>
double measureTimePerElementWithRepsAndWarmup(std::size_t num,
                                              InputDistribution distribution) {
  std::size_t numberOfTests = std::max<std::size_t>(1, (1 << 22) / num);
  std::size_t numberOfWarmups = std::max<std::size_t>(1, (1 << 18) / num);

  InputData<K, Ps...> **data =
      new InputData<K, Ps...> *[numberOfTests + numberOfWarmups];
  for (std::size_t i = 0; i < numberOfTests + numberOfWarmups; i++) {
    data[i] = new InputData<K, Ps...>(num, distribution);
  }
  for (std::size_t j = 0; j < numberOfWarmups; j++) {
    measureTimePerElement<SortMethod, K, Ps...>(*data[j]);
  }
  double sum = 0;
  for (std::size_t j = 0; j < numberOfTests; j++) {
    sum +=
        measureTimePerElement<SortMethod, K, Ps...>(*data[j + numberOfWarmups]);
  }
  for (std::size_t i = 0; i < numberOfTests + numberOfWarmups; i++) {
    delete data[i];
  }
  delete[] data;
  return sum / numberOfTests;
}

template <typename SortMethod, typename K, typename... Ps>
double measureTimePerElementThresh(const InputData<K, Ps...> &data,
                                   radixSort::SortIndex cmpSortThreshold) {
  InputData<K, Ps...> copyOfData(data);
  static_assert(SortMethod::hasThreshold, "SortMethod does not have threshold");
  struct timespec start, end;
  if constexpr (SortMethod::areKeyAndPayloadSeparate) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    std::apply(
        [&](auto *...payloads) {
          SortMethod::template sortThresh(cmpSortThreshold, copyOfData.num,
                                          copyOfData.keys, payloads...);
        },
        copyOfData.payloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
  } else {
    InputDataElement<K, Ps...> *keysAndPayloads =
        (InputDataElement<K, Ps...> *)malloc(
            copyOfData.num * sizeof(InputDataElement<K, Ps...>));
    copyOfData.convertToSingleArray(keysAndPayloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    SortMethod::template sortThresh<K, Ps...>(cmpSortThreshold, copyOfData.num,
                                              keysAndPayloads);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    copyOfData.setFromSingleArray(keysAndPayloads);
    free(keysAndPayloads);
  }
  if (SortMethod::name().find("NoCmp") == std::string::npos &&
      !copyOfData.isSorted(true)) {
    std::cout << "ERROR: "
              << "not sorted (" + std::to_string(copyOfData.numUnsorted(true)) +
                     " out of " + std::to_string(copyOfData.num - 1) + " pairs)"
              << "(" << data.num << " " << SortMethod::name() << ")"
              << std::endl;
    exit(-1);
  }
  return ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) /
         (double)data.num;
}

template <typename SortMethod, typename K, typename... Ps>
double measureTimePerElementThreshWithRepsAndWarmup(
    std::size_t num, InputDistribution distribution,
    radixSort::SortIndex cmpSortThreshold) {
  std::size_t numberOfTests = std::max<std::size_t>(1, (1 << 22) / num);
  std::size_t numberOfWarmups = std::max<std::size_t>(1, (1 << 18) / num);

  InputData<K, Ps...> **data =
      new InputData<K, Ps...> *[numberOfTests + numberOfWarmups];
  for (std::size_t i = 0; i < numberOfTests + numberOfWarmups; i++) {
    data[i] = new InputData<K, Ps...>(num, distribution);
  }
  for (std::size_t j = 0; j < numberOfWarmups; j++) {
    measureTimePerElementThresh<SortMethod, K, Ps...>(*data[j],
                                                      cmpSortThreshold);
  }
  double sum = 0;
  for (std::size_t j = 0; j < numberOfTests; j++) {
    sum += measureTimePerElementThresh<SortMethod, K, Ps...>(
        *data[j + numberOfWarmups], cmpSortThreshold);
  }
  for (std::size_t i = 0; i < numberOfTests + numberOfWarmups; i++) {
    delete data[i];
  }
  delete[] data;
  return sum / numberOfTests;
}

template <typename SortMethod, InputDistribution Distribution, typename K,
          typename... Ps>
void perfTestThresh() {
  system(("mkdir -p " + dataDir).c_str());
  std::string perfDescription = std::string("cmpThresh-") + type_name<K>;
  ((perfDescription += std::string("-") + type_name<Ps>), ...);
  perfDescription += std::string("-") + SortMethod::name() + "-" +
                     inputDistributionToString<Distribution>();
  std::ofstream file(dataDir + "/" + perfDescription + ".dat");
  file << std::fixed;
  file << std::setprecision(6);
  std::cout << "Performing perf test for " << perfDescription << " ..."
            << std::endl;
  const int num = 1 << 18;

  file << "cmpThresh"
       << " " << num << std::endl;
  const radixSort::SortIndex maxThresh =
      std::is_same_v<SortMethod,
                     SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                         radixSort::CmpSorterNoSort>>
          ? num
          : 512;
  const radixSort::SortIndex minThresh =
      std::is_same_v<SortMethod,
                     SortMethodRadixSort<radixSort::BitSorterSIMD<false>,
                                         radixSort::CmpSorterNoSort>>
          ? 1
          : 2;
  for (radixSort::SortIndex thresh = minThresh; thresh <= maxThresh;
       thresh *= 2) {
    file << thresh;
    if (std::is_same_v<SortMethod, SortMethodBramas> &&
        (Distribution == InputDistribution::Zero ||
         Distribution == InputDistribution::ZeroOne ||
         Distribution == InputDistribution::Sorted) &&
        num > 1 << 18) {
      // bramas uses quicksort wich has a runtime of O(n^2) for constant
      // input, so only execute tests of the number of elements is small
      // enough to avoid taking too long
      file << " " << 0;
      continue;
    }
    file << " "
         << measureTimePerElementThreshWithRepsAndWarmup<SortMethod, K, Ps...>(
                num, Distribution, thresh);
    file << std::endl;
  }

  file.close();
  std::cout << "Perf test for " << perfDescription << " done." << std::endl;
}

template <typename SortMethod, typename SortMethodRelTo,
          InputDistribution Distribution, int Payloads, typename K,
          typename... Ps>
static std::string perfTestSpeedup(const size_t num) {
  std::string result = "";
  InputData<K, Ps...> data(num, Distribution);
  if ((std::is_same_v<SortMethodRelTo, SortMethodBramas> ||
       std::is_same_v<SortMethod,
                      SortMethodBramas>)&&(Distribution ==
                                               InputDistribution::Zero ||
                                           Distribution ==
                                               InputDistribution::ZeroOne ||
                                           Distribution ==
                                               InputDistribution::Sorted) &&
      num > 1 << 18) {
    // bramas uses quicksort wich has a runtime of O(n^2) for constant
    // input, so only execute tests of the number of elements is small
    // enough to avoid taking too long
    return "";
  }
  if constexpr (!SortMethodRelTo::template isSupported<K, Ps...>() ||
                !SortMethod::template isSupported<K, Ps...>()) {
    return "";
  }
  double sumTimePerElementRel =
      measureTimePerElementWithRepsAndWarmup<SortMethodRelTo, K, Ps...>(
          num, Distribution);
  double sumTimePerElement =
      measureTimePerElementWithRepsAndWarmup<SortMethod, K, Ps...>(
          num, Distribution);
  result += type_name<K>;
  if constexpr (Payloads == 1) {
    ((result += std::string("-") + type_name<Ps>), ...);
  }
  result += " ";
  result += std::to_string(sumTimePerElementRel / sumTimePerElement);
  result += "\n";
  return result;
}

template <typename SortMethod, typename SortMethodRelTo,
          InputDistribution Distribution, bool WithoutPayload, int Payloads,
          int PayloadFactor = 1>
static void perfTestSpeedupAllKP() {
  const std::size_t num = 1 << 18;
  system(("mkdir -p " + dataDir).c_str());
  std::string perfDescription = std::string() + "relTo" +
                                SortMethodRelTo::name() + "-" +
                                SortMethod::name() + "-";
  if constexpr (WithoutPayload) {
    perfDescription += "withoutPayload";
  } else {
    perfDescription += "withPayloadFactor" + std::to_string(PayloadFactor);
  }
  perfDescription += std::string("-") +
                     inputDistributionToString<Distribution>() +
                     std::string("-") + std::to_string(num);
  std::ofstream file(dataDir + "/" + perfDescription + ".dat");
  file << std::fixed;
  file << std::setprecision(6);
  std::cout << "Performing perf test for " << perfDescription << " ..."
            << std::endl;
  file << "key_payloads speedup" << std::endl;
  if constexpr (WithoutPayload) {
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            float>(num);
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            double>(num);
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            int8_t>(num);
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            int16_t>(num);
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            int32_t>(num);
    file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution, Payloads,
                            int64_t>(num);
  } else {
    if constexpr (Payloads == 0) {
      if constexpr (PayloadFactor == 1) {
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, float, int32_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, double, int64_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int8_t, int8_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int16_t, int16_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int32_t, int32_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int64_t, int64_t>(num);
      } else if constexpr (PayloadFactor == 2) {
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, float, int16_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, double, int32_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int16_t, int8_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int32_t, int16_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int64_t, int32_t>(num);
      } else if constexpr (PayloadFactor == 4) {
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, float, int8_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, double, int16_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int32_t, int8_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int64_t, int16_t>(num);
      } else if constexpr (PayloadFactor == 8) {
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, double, int8_t>(num);
        file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                                Payloads, int64_t, int8_t>(num);
      }
    } else if constexpr (Payloads == 1) {
      file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                              Payloads, float, int64_t>(num);
      file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                              Payloads, int8_t, int64_t>(num);
      file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                              Payloads, int16_t, int64_t>(num);
      file << perfTestSpeedup<SortMethod, SortMethodRelTo, Distribution,
                              Payloads, int32_t, int64_t>(num);
    }
  }

  file.close();
  std::cout << "Perf test for " << perfDescription << " done." << std::endl;
}

template <typename SortMethod, typename SortMethodRelTo,
          InputDistribution Distribution, int Payloads = 0>
static void perfTestSpeedupAll() {
  perfTestSpeedupAllKP<SortMethod, SortMethodRelTo, Distribution, true,
                       Payloads, 0>();
  perfTestSpeedupAllKP<SortMethod, SortMethodRelTo, Distribution, false,
                       Payloads, 1>();
  perfTestSpeedupAllKP<SortMethod, SortMethodRelTo, Distribution, false,
                       Payloads, 2>();
  perfTestSpeedupAllKP<SortMethod, SortMethodRelTo, Distribution, false,
                       Payloads, 4>();
  perfTestSpeedupAllKP<SortMethod, SortMethodRelTo, Distribution, false,
                       Payloads, 8>();
}

template <typename... SortMethods> struct PerfTest {
  static_assert(sizeof...(SortMethods) > 0, "No sort methods provided");

  template <InputDistribution Distribution, typename K, typename... Ps>
  static void perfTestNum() {
    system(("mkdir -p " + dataDir).c_str());
    std::string perfDescription = std::string("tpe-") + type_name<K>;
    ((perfDescription += std::string("-") + type_name<Ps>), ...);
    perfDescription +=
        std::string("-") + inputDistributionToString<Distribution>();
    std::ofstream file(dataDir + "/" + perfDescription + ".dat");
    file << std::fixed;
    file << std::setprecision(6);
    std::cout << "Performing perf test for " << perfDescription << " ..."
              << std::endl;
    file << "number_of_elements";
    ((file << " " << SortMethods::name()), ...);
    file << std::endl;
    std::size_t maxNum = 1 << 22;
    for (std::size_t num = 1; num <= maxNum; num *= 2) {
      file << num;
      (
          [&]() {
            if (std::is_same_v<SortMethods, SortMethodBramas> &&
                (Distribution == InputDistribution::Zero ||
                 Distribution == InputDistribution::ZeroOne ||
                 Distribution == InputDistribution::Sorted) &&
                num > 1 << 18) {
              // bramas uses quicksort wich has a runtime of O(n^2) for constant
              // input, so only execute tests of the number of elements is small
              // enough to avoid taking too long
              file << " 0";
              return;
            }
            if constexpr (SortMethods::template isSupported<K, Ps...>()) {
              file << " "
                   << measureTimePerElementWithRepsAndWarmup<SortMethods, K,
                                                             Ps...>(
                          num, Distribution);
            } else {
              file << " nan";
            }
          }(),
          ...);
      file << std::endl;
    }
    file.close();
    std::cout << "Perf test for " << perfDescription << " done." << std::endl;
  }

  template <InputDistribution Distribution, typename K, typename... Ps>
  static void perfTest() {
    const std::size_t num = 1 << 18;
    system(("mkdir -p " + dataDir).c_str());
    std::string perfDescription = std::string() + type_name<K>;
    ((perfDescription += std::string("-") + type_name<Ps>), ...);
    perfDescription += std::string("-") +
                       inputDistributionToString<Distribution>() +
                       std::string("-") + std::to_string(num);
    std::ofstream file(dataDir + "/" + perfDescription + ".dat");
    std::cout << "Performing perf test for " << perfDescription << " ..."
              << std::endl;
    file << std::fixed;
    file << std::setprecision(6);
    file << "sort_method nanoseconds_per_element" << std::endl;
    (
        [&]() {
          if (std::is_same_v<SortMethods, SortMethodBramas> &&
              (Distribution == InputDistribution::Zero ||
               Distribution == InputDistribution::ZeroOne ||
               Distribution == InputDistribution::Sorted) &&
              num > 1 << 18) {
            // bramas uses quicksort wich has a runtime of O(n^2) for constant
            // input, so only execute tests of the number of elements is small
            // enough to avoid taking too long
            file << SortMethods::name() << " " << 0 << std::endl;
            return;
          }
          if constexpr (SortMethods::template isSupported<K, Ps...>()) {
            file << SortMethods::name() << " "
                 << measureTimePerElementWithRepsAndWarmup<SortMethods, K,
                                                           Ps...>(num,
                                                                  Distribution)
                 << std::endl;
          }
        }(),
        ...);
    file << std::endl;
    file.close();
    std::cout << "Perf test for " << perfDescription << " done." << std::endl;
  }

  template <InputDistribution Distribution, typename K, typename... Ps>
  static void thresh() {
    (
        [&]() {
          if constexpr (SortMethods::hasThreshold &&
                        SortMethods::template isSupported<K, Ps...>()) {
            perfTestThresh<SortMethods, Distribution, K, Ps...>();
          }
        }(),
        ...);
  }
};

int main(int argc, char const *argv[]) {
  std::cout << std::fixed;
  std::cout << std::setprecision(6);
#ifdef _IPP_RADIX_IS_PRESENT_
  std::cout << "Initializing IPP..." << std::endl;
  ippRadix::ippInit();
#endif
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
#endif
#ifdef _IPP_RADIX_IS_PRESENT_
               SortMethodIPPRadix,
#endif
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
#endif

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
