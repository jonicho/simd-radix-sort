#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <popcntintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common.hpp"

namespace simd_sort::simd {
// simd library inspired by
// the T-SIMD library written by Prof. Dr.-Ing. Ralf Möller:
// https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html

namespace internal {
template <typename T, std::size_t Bytes, typename = void>
struct MMRegType {
  static_assert(always_false_v<T>, "Unsupported type or number of bytes");
};
template <typename T>
struct MMRegType<T, 64> {
  using type = __m512i;
};
template <typename T>
struct MMRegType<T, 32> {
  using type = __m256i;
};
template <typename T>
struct MMRegType<T, 16> {
  using type = __m128i;
};
template <typename T>
struct MMRegType<T, 8> {
  using type = __m128i;
};
}  // namespace internal

template <typename T, std::size_t Bytes>
using MMRegType = typename internal::MMRegType<T, Bytes>::type;

template <typename T, std::size_t Bytes = 64, typename = void>
struct Vec;

template <typename T, std::size_t Bytes>
struct Vec<T, Bytes,
           std::enable_if_t<(Bytes <= 64) && is_power_of_two<Bytes>>> {
  MMRegType<T, Bytes> mmReg;
  static constexpr std::size_t numElems = Bytes / sizeof(T);
  Vec() = default;
  Vec(const MMRegType<T, Bytes> x) : mmReg(x) {}
  Vec &operator=(const MMRegType<T, Bytes> x) {
    mmReg = x;
    return *this;
  }
  operator MMRegType<T, Bytes>() const { return mmReg; }
};

template <typename T, std::size_t Bytes>
struct Vec<T, Bytes, std::enable_if_t<(Bytes > 64) && is_power_of_two<Bytes>>> {
  MMRegType<T, 64> mmReg[Bytes / 64];
  static constexpr std::size_t numElems = Bytes / sizeof(T);
  static constexpr std::size_t numRegs = Bytes / 64;
  Vec() = default;
  MMRegType<T, 64> &operator[](std::size_t i) { return mmReg[i]; }
  const MMRegType<T, 64> &operator[](std::size_t i) const { return mmReg[i]; }
};

template <std::size_t Size>
struct _MaskType {
  static_assert(always_false_v<_MaskType<Size>>, "Unsupported mask size");
};
template <>
struct _MaskType<64> {
  using type = __mmask64;
};
template <>
struct _MaskType<32> {
  using type = __mmask32;
};
template <>
struct _MaskType<16> {
  using type = __mmask16;
};
template <>
struct _MaskType<8> {
  using type = __mmask8;
};
template <>
struct _MaskType<4> {
  using type = __mmask8;
};
template <>
struct _MaskType<2> {
  using type = __mmask8;
};
template <>
struct _MaskType<1> {
  using type = __mmask8;
};

template <std::size_t Size>
using MaskType = typename _MaskType<Size>::type;

template <std::size_t Size>
struct Mask {
  MaskType<Size> k;
  Mask() = default;
  Mask(const MaskType<Size> &x) : k(x) {}
  Mask &operator=(const MaskType<Size> &x) {
    k = x;
    return *this;
  }
  operator MaskType<Size>() const { return k; }
};

template <typename Tdst, typename Tsrc, std::size_t Bytes>
static INLINE Vec<Tdst, Bytes> reinterpret(const Vec<Tsrc, Bytes> &vec) {
  return reinterpret_cast<const Vec<Tdst, Bytes> &>(vec);
}

template <typename T, std::size_t Bytes = 64>
static INLINE Vec<T, Bytes> set_bit(const std::size_t bitNo) {
  if constexpr (Bytes == 64) {
    if constexpr (sizeof(T) == 1) {
      return _mm512_set1_epi8(uint8_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 2) {
      return _mm512_set1_epi16(uint16_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 4) {
      return _mm512_set1_epi32(uint32_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 8) {
      return _mm512_set1_epi64(uint64_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 16) {
      return _mm512_maskz_set1_epi64(0b01010101 << (bitNo / 64),
                                     uint64_t(1) << (bitNo % 64));
    } else if constexpr (sizeof(T) == 32) {
      return _mm512_maskz_set1_epi64(0b00010001 << (bitNo / 64),
                                     uint64_t(1) << (bitNo % 64));
    } else if constexpr (sizeof(T) == 64) {
      return _mm512_maskz_set1_epi64(0b00000001 << (bitNo / 64),
                                     uint64_t(1) << (bitNo % 64));
    } else {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
  } else if constexpr (Bytes == 32) {
    if constexpr (sizeof(T) == 1) {
      return _mm256_set1_epi8(uint8_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 2) {
      return _mm256_set1_epi16(uint16_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 4) {
      return _mm256_set1_epi32(uint32_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 8) {
      return _mm256_set1_epi64x(uint64_t(1) << bitNo);
#ifdef __AVX512VL__
    } else if constexpr (sizeof(T) == 16) {
      return _mm256_maskz_set1_epi64(0b0101 << (bitNo / 64),
                                     uint64_t(1) << (bitNo % 64));
    } else if constexpr (sizeof(T) == 32) {
      return _mm256_maskz_set1_epi64(0b0001 << (bitNo / 64),
                                     uint64_t(1) << (bitNo % 64));
#endif  // __AVX512VL__
    } else {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
  } else if constexpr (Bytes == 16 || Bytes == 8) {
    if constexpr (sizeof(T) == 1) {
      return _mm_set1_epi8(uint8_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 2) {
      return _mm_set1_epi16(uint16_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 4) {
      return _mm_set1_epi32(uint32_t(1) << bitNo);
    } else if constexpr (sizeof(T) == 8) {
      return _mm_set1_epi64x(uint64_t(1) << bitNo);
#ifdef __AVX512VL__
    } else if constexpr (sizeof(T) == 16) {
      return _mm_maskz_set1_epi64(0b01 << (bitNo / 64), uint64_t(1)
                                                            << (bitNo % 64));
#endif  // __AVX512VL__
    } else {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <std::size_t Bytes = 64, typename T>
static INLINE Vec<T, Bytes> loadu(const T *p) {
  if constexpr (Bytes == 64) {
    return _mm512_loadu_si512(p);
  } else if constexpr (Bytes == 32) {
    return _mm256_loadu_si256((__m256i_u *)p);
  } else if constexpr (Bytes == 16) {
    return _mm_loadu_si128((__m128i_u *)p);
#if defined(__AVX512BW__) && defined(__AVX512VL__)
  } else if constexpr (Bytes == 8) {
    if constexpr (sizeof(T) == 1) {
      return _mm_maskz_loadu_epi8(0xff, p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
#endif  // __AVX512BW__ && __AVX512VL__
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Vec<T, Bytes> result;
    for (std::size_t i = 0; i < Vec<T, Bytes>::numRegs; i++) {
      result[i] = _mm512_loadu_si512(p + i * Vec<T, 64>::numElems);
    }
    return result;
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <std::size_t Bytes = 64, typename T>
static INLINE Vec<T, Bytes> maskz_loadu(const Mask<Vec<T, Bytes>::numElems> m,
                                        const T *p) {
  if constexpr (Bytes == 64) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm512_maskz_loadu_epi8(m, p);
    } else if constexpr (sizeof(T) == 2) {
      return _mm512_maskz_loadu_epi16(m, p);
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm512_maskz_loadu_epi32(m, p);
      } else if constexpr (sizeof(T) <= 64) {
        return _mm512_maskz_loadu_epi64(m, p);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm256_maskz_loadu_epi8(m, (__m256i_u *)p);
    } else if constexpr (sizeof(T) == 2) {
      return _mm256_maskz_loadu_epi16(m, (__m256i_u *)p);
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm256_maskz_loadu_epi32(m, (__m256i_u *)p);
      } else if constexpr (sizeof(T) <= 32) {
        return _mm256_maskz_loadu_epi64(m, (__m256i_u *)p);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
  } else if constexpr (Bytes == 16) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm_maskz_loadu_epi8(m, p);
    } else if constexpr (sizeof(T) == 2) {
      return _mm_maskz_loadu_epi16(m, p);
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm_maskz_loadu_epi32(m, p);
      } else if constexpr (sizeof(T) <= 16) {
        return _mm_maskz_loadu_epi64(m, p);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
  } else if constexpr (Bytes == 8) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm_maskz_loadu_epi8(m & 0xff, p);
    } else
#endif  // __AVX512BW__
    {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
#endif  // __AVX512VL__
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Vec<T, Bytes> result;
    Mask<Vec<T, Bytes>::numElems> mask = m;
    for (std::size_t i = 0; i < Vec<T, Bytes>::numRegs; i++) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        result[i] = _mm512_maskz_loadu_epi8(mask, p + i * Vec<T, 64>::numElems);
      } else if constexpr (sizeof(T) == 2) {
        result[i] =
            _mm512_maskz_loadu_epi16(mask, p + i * Vec<T, 64>::numElems);
      } else
#endif  // __AVX512BW__
        if constexpr (sizeof(T) == 4) {
          result[i] =
              _mm512_maskz_loadu_epi32(mask, p + i * Vec<T, 64>::numElems);
        } else if constexpr (sizeof(T) <= 64) {
          result[i] =
              _mm512_maskz_loadu_epi64(mask, p + i * Vec<T, 64>::numElems);
        } else {
          static_assert(always_false_v<T>, "Unsupported type size");
        }
      mask = kshiftr(mask, Vec<T, 64>::numElems);
    }
    return result;
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <typename T, std::size_t Bytes>
static INLINE void mask_compressstoreu(T *p,
                                       const Mask<Vec<T, Bytes>::numElems> m,
                                       const Vec<T, Bytes> v) {
  if constexpr (Bytes == 64) {
#ifdef __AVX512VBMI2__
    if constexpr (sizeof(T) == 1) {
      _mm512_mask_compressstoreu_epi8(p, m, v);
    } else if constexpr (sizeof(T) == 2) {
      _mm512_mask_compressstoreu_epi16(p, m, v);
    } else
#endif  // __AVX512VBMI2__
      if constexpr (sizeof(T) == 4) {
        _mm512_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) <= 64) {
        _mm512_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
#ifdef __AVX512VBMI2__
    if constexpr (sizeof(T) == 1) {
      _mm256_mask_compressstoreu_epi8(p, m, v);
    } else if constexpr (sizeof(T) == 2) {
      _mm256_mask_compressstoreu_epi16(p, m, v);
    } else
#endif  // __AVX512VBMI2__
      if constexpr (sizeof(T) == 4) {
        _mm256_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) <= 32) {
        _mm256_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
  } else if constexpr (Bytes == 16) {
#ifdef __AVX512VBMI2__
    if constexpr (sizeof(T) == 1) {
      _mm_mask_compressstoreu_epi8(p, m, v);
    } else if constexpr (sizeof(T) == 2) {
      _mm_mask_compressstoreu_epi16(p, m, v);
    } else
#endif  // __AVX512VBMI2__
      if constexpr (sizeof(T) == 4) {
        _mm_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) <= 16) {
        _mm_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
  } else if constexpr (Bytes == 8) {
#ifdef __AVX512VBMI2__
    if constexpr (sizeof(T) == 1) {
      _mm_mask_compressstoreu_epi8(p, m & 0xff, v);
    } else
#endif  // __AVX512VBMI2__
    {
      static_assert(always_false_v<T>, "Unsupported type size");
    }
#endif  // __AVX512VL__
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Mask<Vec<T, Bytes>::numElems> mask = m;
    for (std::size_t i = 0; i < Vec<T, Bytes>::numRegs; i++) {
      if constexpr (sizeof(T) == 2) {
        _mm512_mask_compressstoreu_epi16(p, mask, v[i]);
      } else if constexpr (sizeof(T) == 4) {
        _mm512_mask_compressstoreu_epi32(p, mask, v[i]);
      } else if constexpr (sizeof(T) <= 64) {
        _mm512_mask_compressstoreu_epi64(p, mask, v[i]);
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
      p += kpopcnt((Mask<Vec<T, 64>::numElems>)mask);
      mask = kshiftr(mask, Vec<T, 64>::numElems);
    }
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <typename T, std::size_t Bytes>
static INLINE Mask<Vec<T, Bytes>::numElems> test_bit(const Vec<T, Bytes> v,
                                                     const std::size_t bitNo) {
  if constexpr (Bytes == 64) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm512_test_epi8_mask(v, set_bit<T, Bytes>(bitNo));
    } else if constexpr (sizeof(T) == 2) {
      return _mm512_test_epi16_mask(v, set_bit<T, Bytes>(bitNo));
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm512_test_epi32_mask(v, set_bit<T, Bytes>(bitNo));
      } else if constexpr (sizeof(T) == 8) {
        return _mm512_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
      } else if constexpr (sizeof(T) == 16) {
        __mmask8 mask = _mm512_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
        if (bitNo < 64) {
          // mask is now 0b0a 0b 0c 0d, we want 0baa bb cc dd
          return mask | (mask << 1);
        } else {
          // mask is now 0ba0 b0 c0 d0, we want 0baa bb cc dd
          return mask | (mask >> 1);
        }
      } else if constexpr (sizeof(T) == 32) {
        __mmask8 mask = _mm512_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
        if (bitNo < 64) {
          // mask is now 0b000a 000b, we want 0baaaa bbbb
          return mask | (mask << 1) | (mask << 2) | (mask << 3);
        } else if (bitNo < 128) {
          // mask is now 0b00a0 00b0, we want 0baaaa bbbb
          return (mask >> 1) | mask | (mask << 1) | (mask << 2);
        } else if (bitNo < 192) {
          // mask is now 0b0a00 0b00, we want 0baaaa bbbb
          return (mask >> 2) | (mask >> 1) | mask | (mask << 1);
        } else {
          // mask is now 0ba000 b000, we want 0baaaa bbbb
          return (mask >> 3) | (mask >> 2) | (mask >> 1) | mask;
        }
      } else if constexpr (sizeof(T) == 64) {
        __mmask8 mask = _mm512_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
        // mask is now 0b0000 000a, or 0b0000 00a0, or ...
        // we want 0baaaa aaaa
        return mask == 0 ? 0 : 0xff;
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm256_test_epi8_mask(v, set_bit<T, Bytes>(bitNo));
    } else if constexpr (sizeof(T) == 2) {
      return _mm256_test_epi16_mask(v, set_bit<T, Bytes>(bitNo));
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm256_test_epi32_mask(v, set_bit<T, Bytes>(bitNo));
      } else if constexpr (sizeof(T) == 8) {
        return _mm256_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
  } else if constexpr (Bytes == 16 || Bytes == 8) {
#ifdef __AVX512BW__
    if constexpr (sizeof(T) == 1) {
      return _mm_test_epi8_mask(v, set_bit<T, Bytes>(bitNo));
    } else if constexpr (sizeof(T) == 2) {
      return _mm_test_epi16_mask(v, set_bit<T, Bytes>(bitNo));
    } else
#endif  // __AVX512BW__
      if constexpr (sizeof(T) == 4) {
        return _mm_test_epi32_mask(v, set_bit<T, Bytes>(bitNo));
      } else if constexpr (sizeof(T) == 8) {
        return _mm_test_epi64_mask(v, set_bit<T, Bytes>(bitNo));
      } else {
        static_assert(always_false_v<T>, "Unsupported type size");
      }
#endif  // __AVX512VL__
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <std::size_t Size>
static INLINE std::size_t kpopcnt(const Mask<Size> m) {
  if constexpr (Size < 8) {
    return _mm_popcnt_u64(m) / (8 / Size);
  } else {
    return _mm_popcnt_u64(m);
  }
}

template <std::size_t Size>
static INLINE Mask<Size> knot(const Mask<Size> m) {
#ifdef __AVX512DQ__
  if constexpr (Size <= 8) {
    return _knot_mask8(m);
  } else
#endif  // __AVX512DQ__
    if constexpr (Size == 16) {
      return _knot_mask16(m);
#ifdef __AVX512BW__
    } else if constexpr (Size == 32) {
      return _knot_mask32(m);
    } else if constexpr (Size == 64) {
      return _knot_mask64(m);
#endif  // __AVX512BW__
    } else {
      static_assert(always_false_v<Mask<Size>>, "Unsupported mask size");
    }
}

template <std::size_t Size>
static INLINE Mask<Size> kand(const Mask<Size> m1, const Mask<Size> m2) {
#ifdef __AVX512DQ__
  if constexpr (Size <= 8) {
    return _kand_mask8(m1, m2);
  } else
#endif  // __AVX512DQ__
    if constexpr (Size == 16) {
      return _kand_mask16(m1, m2);
#ifdef __AVX512BW__
    } else if constexpr (Size == 32) {
      return _kand_mask32(m1, m2);
    } else if constexpr (Size == 64) {
      return _kand_mask64(m1, m2);
#endif  // __AVX512BW__
    } else {
      static_assert(always_false_v<Mask<Size>>, "Unsupported mask size");
    }
}

template <std::size_t Size>
static INLINE Mask<Size> kshiftr(const Mask<Size> m, const std::size_t n) {
  if constexpr (Size <= 8) {
    return m >> (n * (8 / Size));
  } else {
    return m >> n;
  }
}

template <std::size_t Size>
static INLINE Mask<Size> kshiftl(const Mask<Size> m, const std::size_t n) {
  if constexpr (Size <= 8) {
    return m << (n * (8 / Size));
  } else {
    return m << n;
  }
}
}  // namespace simd_sort::simd
