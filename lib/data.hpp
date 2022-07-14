#ifndef _DATA_H_
#define _DATA_H_

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

template <typename T> T getRandom() {
  uint32_t randomInts[std::max<int>(sizeof(T) / sizeof(uint32_t), 1)];
  for (std::size_t i = 0; i < sizeof(randomInts) / sizeof(uint32_t); i++) {
    randomInts[i] = rand();
  }
  T result;
  memcpy(&result, randomInts, sizeof(T));
  return result;
}

enum class InputDistribution {
  Uniform,
  Gaussian,
  Zero,
  ZeroOne,
  Sorted,
  ReverseSorted,
  AlmostSorted,
  AlmostReverseSorted,
};

template <InputDistribution Distribution>
constexpr const char *inputDistributionToString() {
  if constexpr (Distribution == InputDistribution::Zero) {
    return "Zero";
  } else if constexpr (Distribution == InputDistribution::ZeroOne) {
    return "ZeroOne";
  } else if constexpr (Distribution == InputDistribution::Uniform) {
    return "Uniform";
  } else if constexpr (Distribution == InputDistribution::Gaussian) {
    return "Gaussian";
  } else if constexpr (Distribution == InputDistribution::Sorted) {
    return "Sorted";
  } else if constexpr (Distribution == InputDistribution::ReverseSorted) {
    return "ReverseSorted";
  } else if constexpr (Distribution == InputDistribution::AlmostSorted) {
    return "AlmostSorted";
  } else if constexpr (Distribution == InputDistribution::AlmostReverseSorted) {
    return "AlmostReverseSorted";
  } else {
    return "Unknown";
  }
}

template <typename K, typename... Ps> struct InputDataElement {
  K key;
  std::tuple<Ps...> payloads;

  bool operator<(const InputDataElement &other) const {
    return key < other.key;
  }
  bool operator>(const InputDataElement &other) const {
    return key > other.key;
  }
  bool operator<=(const InputDataElement &other) const {
    return key <= other.key;
  }
  bool operator>=(const InputDataElement &other) const {
    return key >= other.key;
  }
  bool operator==(const InputDataElement &other) const {
    return key == other.key;
  }
  bool operator!=(const InputDataElement &other) const {
    return key != other.key;
  }
};

// specialization for of InputDataElement for no payloads, because an empty
// tuple still uses one byte of space, we don't want that
template <typename K> struct InputDataElement<K> {
  K key;

  bool operator<(const InputDataElement &other) const {
    return key < other.key;
  }
  bool operator>(const InputDataElement &other) const {
    return key > other.key;
  }
  bool operator<=(const InputDataElement &other) const {
    return key <= other.key;
  }
  bool operator>=(const InputDataElement &other) const {
    return key >= other.key;
  }
  bool operator==(const InputDataElement &other) const {
    return key == other.key;
  }
  bool operator!=(const InputDataElement &other) const {
    return key != other.key;
  }
};

template <typename K, typename... Ps> struct InputData {
  K *keys;
  std::tuple<Ps *...> payloads;
  std::size_t num;

  InputData(std::size_t num, InputDistribution distribution,
            uint seed = time(NULL))
      : num(num) {
    keys = new K[num];
    payloads = std::make_tuple(new Ps[num]...);
    std::mt19937 gen(seed);
    switch (distribution) {
    case InputDistribution::Zero: {
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = 0;
      }
    } break;
    case InputDistribution::ZeroOne: {
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = K(gen() % 2);
      }
    } break;
    case InputDistribution::Uniform: {
      fillKeysUniform(gen);
    } break;
    case InputDistribution::Gaussian: {
      fillKeysGaussian(gen);
    } break;
    case InputDistribution::Sorted: {
      if constexpr (std::is_integral_v<K>) {
        fillKeysUniform(gen);
      } else {
        fillKeysGaussian(gen);
      }
      std::sort(keys, keys + num);
    } break;
    case InputDistribution::ReverseSorted: {
      if constexpr (std::is_integral_v<K>) {
        fillKeysUniform(gen);
      } else {
        fillKeysGaussian(gen);
      }
      std::sort(keys, keys + num, std::greater<K>());
    } break;
    case InputDistribution::AlmostSorted: {
      if constexpr (std::is_integral_v<K>) {
        fillKeysUniform(gen);
      } else {
        fillKeysGaussian(gen);
      }
      std::sort(keys, keys + num);
      std::size_t numberOfDisplacements = std::exp2(std::log10(num));
      std::uniform_int_distribution<std::size_t> dist(0, num - 1);
      for (std::size_t i = 0; i < numberOfDisplacements; i++) {
        std::swap(keys[dist(gen)], keys[dist(gen)]);
      }
    } break;
    case InputDistribution::AlmostReverseSorted: {
      if constexpr (std::is_integral_v<K>) {
        fillKeysUniform(gen);
      } else {
        fillKeysGaussian(gen);
      }
      std::sort(keys, keys + num, std::greater<K>());
      std::size_t numberOfDisplacements = std::exp2(std::log10(num));
      std::uniform_int_distribution<std::size_t> dist(0, num - 1);
      for (std::size_t i = 0; i < numberOfDisplacements; i++) {
        std::swap(keys[dist(gen)], keys[dist(gen)]);
      }
    } break;
    }
    makePayloads();
  }

  InputData(const InputData &other) : num(other.num) {
    keys = new K[num];
    payloads = std::make_tuple(new Ps[num]...);
    std::copy(other.keys, other.keys + num, keys);
    std::apply(
        [&](auto... newPayloads) {
          std::apply(
              [&](auto... otherPayloads) {
                (std::copy(otherPayloads, otherPayloads + num, newPayloads),
                 ...);
              },
              other.payloads);
        },
        payloads);
  }

  ~InputData() {
    delete[] keys;
    std::apply([](Ps *...payloads) { (..., delete[] payloads); }, payloads);
  }

  bool isSorted(bool up = true) const {
    if (up) {
      for (std::size_t i = 1; i < num; ++i) {
        if (keys[i - 1] > keys[i]) {
          // std::cout << "keys[" << i - 1
          //           << "] = " << std::bitset<sizeof(K) * 8>(keys[i - 1])
          //           << " > keys[" << i
          //           << "] = " << std::bitset<sizeof(K) * 8>(keys[i])
          //           << std::endl;
          return false;
        }
      }
    } else {
      for (std::size_t i = 1; i < num; ++i) {
        if (keys[i - 1] < keys[i]) {
          // std::cout << "keys[" << i - 1
          //           << "] = " << std::bitset<sizeof(K) * 8>(keys[i - 1])
          //           << " < keys[" << i
          //           << "] = " << std::bitset<sizeof(K) * 8>(keys[i])
          //           << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  bool checkThatEveryKeyIsThere(const InputData<K, Ps...> &other) const {
    for (std::size_t i = 0; i < other.num; i++) {
      if (std::find(keys, keys + num, other.keys[i]) == keys + num) {
        return false;
      }
    }
    return true;
  }

  uint numUnsorted(bool up) const {
    uint numUnsorted = 0;
    if (up) {
      for (std::size_t i = 1; i < num; ++i) {
        if (keys[i - 1] > keys[i]) {
          numUnsorted++;
        }
      }
    } else {
      for (std::size_t i = 1; i < num; ++i) {
        if (keys[i - 1] < keys[i]) {
          numUnsorted++;
        }
      }
    }
    return numUnsorted;
  }

  bool checkPayloads() const {
    for (std::size_t i = 0; i < num; ++i) {
      uint seed = 0;
      memcpy(&seed, &keys[i], std::min(sizeof(K), sizeof(uint)));
      srand(seed);
      bool payloadsOk = true;
      auto checkPayload = [&](auto *payload) {
        auto expectedPayload =
            getRandom<std::remove_reference_t<decltype(*payload)>>();
        return memcmp(&payload[i], &expectedPayload, sizeof(payload[i])) == 0;
      };
      std::apply(
          [&](Ps *...payloads) {
            payloadsOk &= (... && checkPayload(payloads));
          },
          payloads);
      if (!payloadsOk) {
        return false;
      }
    }
    return true;
  }

  std::string checkData(bool up, const InputData<K, Ps...> &original,
                        bool checkSorted = true) const {
    bool passed = true;
    std::string errorMessage = "";
    if (checkSorted && !isSorted(up)) {
      passed = false;
      errorMessage += "not sorted (" + std::to_string(numUnsorted(up)) +
                      " out of " + std::to_string(num - 1) + " pairs)";
    }
    if (!checkPayloads()) {
      if (!passed) {
        errorMessage += ", ";
      }
      passed = false;
      errorMessage += "payloads are not ok";
    }
    // std::is_permutation()
    if (!checkThatEveryKeyIsThere(original)) {
      if (!passed) {
        errorMessage += ", ";
      }
      passed = false;
      errorMessage += "keys are not there";
    }
    if (!original.checkThatEveryKeyIsThere(*this)) {
      if (!passed) {
        errorMessage += ", ";
      }
      passed = false;
      errorMessage += "new keys are there";
    }
    return errorMessage;
  }

  void print() const {
    for (std::size_t i = 0; i < num; ++i) {
      std::cout << "(" << +keys[i];
      std::apply(
          [&](Ps *...payloads) { (..., (std::cout << ", " << +payloads[i])); },
          payloads);
      std::cout << "), ";
    }
    std::cout << std::endl;
  }

  void printBinary() const {
    for (std::size_t i = 0; i < num; ++i) {
      std::cout << std::bitset<sizeof(K) * 8>(keys[i]) << " ";
    }
    std::cout << std::endl;
  }

  void convertToSingleArray(InputDataElement<K, Ps...> *keysAndPayloads) const {
    for (std::size_t i = 0; i < num; ++i) {
      keysAndPayloads[i].key = keys[i];
      if constexpr (sizeof...(Ps) > 0) {
        std::apply(
            [&](Ps *...payloads) {
              keysAndPayloads[i].payloads = std::make_tuple(payloads[i]...);
            },
            payloads);
      }
    }
  }

  void setFromSingleArray(InputDataElement<K, Ps...> *keysAndPayloads) {
    for (std::size_t i = 0; i < num; ++i) {
      keys[i] = keysAndPayloads[i].key;
      if constexpr (sizeof...(Ps) > 0) {
        std::apply(
            [&](Ps... payloadsIn) {
              std::apply(
                  [&](Ps *...payloads) { ((payloads[i] = payloadsIn), ...); },
                  payloads);
            },
            keysAndPayloads[i].payloads);
      }
    }
  }

private:
  void fillKeysUniform(std::mt19937 &gen) {
    if constexpr (std::is_integral_v<K>) {
      std::uniform_int_distribution<K> dist(std::numeric_limits<K>::lowest(),
                                            std::numeric_limits<K>::max());
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = dist(gen);
      }
    } else {
      std::uniform_real_distribution<K> dist(K(-1.0), K(1.0));
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = dist(gen);
      }
    }
  }

  void fillKeysGaussian(std::mt19937 &gen) {
    if constexpr (std::is_integral_v<K>) {
      std::normal_distribution<double> dist(0, 100);
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = round(dist(gen));
      }
    } else {
      std::normal_distribution<K> dist(0, 1.0);
      for (std::size_t i = 0; i < num; i++) {
        keys[i] = dist(gen);
      }
    }
  }

  void makePayloads() {
    for (std::size_t i = 0; i < num; ++i) {
      uint seed = 0;
      memcpy(&seed, &keys[i], std::min(sizeof(K), sizeof(uint)));
      srand(seed);
      std::apply(
          [&](auto &...payloads) {
            (...,
             (payloads[i] =
                  getRandom<std::remove_reference_t<decltype(payloads[i])>>()));
          },
          payloads);
    }
  }
};

#endif // _DATA_H_
