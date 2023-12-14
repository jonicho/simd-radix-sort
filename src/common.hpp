#pragma once

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <typeinfo>

namespace simd_sort {

#define INLINE inline __attribute__((always_inline))

template <typename>
inline constexpr bool always_false_v = false;

using SortIndex = ssize_t;

namespace internal {
template <typename T>
struct type_name {
  static constexpr const char *value = typeid(T).name();
};
template <>
struct type_name<bool> {
  static constexpr const char *value = "bool";
};
template <>
struct type_name<uint8_t> {
  static constexpr const char *value = "uint8";
};
template <>
struct type_name<uint16_t> {
  static constexpr const char *value = "uint16";
};
template <>
struct type_name<uint32_t> {
  static constexpr const char *value = "uint32";
};
template <>
struct type_name<uint64_t> {
  static constexpr const char *value = "uint64";
};
template <>
struct type_name<int8_t> {
  static constexpr const char *value = "int8";
};
template <>
struct type_name<int16_t> {
  static constexpr const char *value = "int16";
};
template <>
struct type_name<int32_t> {
  static constexpr const char *value = "int32";
};
template <>
struct type_name<int64_t> {
  static constexpr const char *value = "int64";
};
template <>
struct type_name<float> {
  static constexpr const char *value = "float";
};
template <>
struct type_name<double> {
  static constexpr const char *value = "double";
};
}  // namespace internal

template <typename T>
constexpr const char *type_name = internal::type_name<T>::value;

template <std::size_t Bytes>
struct LargeUInt {
  static_assert(Bytes > 8, "Bytes must be larger than 8");
  static_assert((Bytes & (Bytes - 1)) == 0, "Bytes must be a power of 2");
  uint64_t data[Bytes / 8];

  LargeUInt() = default;
  LargeUInt(const LargeUInt &) = default;
  LargeUInt &operator=(const LargeUInt &) = default;

  LargeUInt(uint64_t x) {
    data[0] = x;
    for (std::size_t i = 1; i < Bytes / 8; i++) {
      data[i] = 0;
    }
  }
};

namespace internal {

template <std::size_t Bytes>
struct UInt;

template <>
struct UInt<1> {
  using type = uint8_t;
};

template <>
struct UInt<2> {
  using type = uint16_t;
};

template <>
struct UInt<4> {
  using type = uint32_t;
};

template <>
struct UInt<8> {
  using type = uint64_t;
};

template <std::size_t Bytes>
struct UInt {
  using type = LargeUInt<Bytes>;
};

}  // namespace internal

template <std::size_t Bytes>
using UInt = typename internal::UInt<Bytes>::type;

template <std::size_t Bytes>
UInt<Bytes> setBit(std::size_t n) {
  if constexpr (Bytes <= 8) {
    return UInt<Bytes>(1ULL << n);
  } else {
    UInt<Bytes> x;
    x.data[n / 64] = 1ULL << (n % 64);
    return x;
  }
}

template <std::size_t X>
static constexpr bool is_power_of_two = X > 0 && (X & (X - 1)) == 0;

}  // namespace simd_sort
