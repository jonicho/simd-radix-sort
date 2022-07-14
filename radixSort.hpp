//=============================================================================
//
// Generic AVX-512 implementation of MSB Radix Sort for sorting datasets
// with a key and an arbitrary number of payloads.
// Header-only library.
//
// By Jonas Richard Keller, 2022.
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
// radixSort::sort(num, keyArray, payloadArrays...);
//
// where num is the number of elements in the dataset, keyArray is a pointer
// to the key array, and payloadArrays are the pointers to the payload arrays.
//
// By default the sort function sorts in ascending order. To sort in descending
// order, call the sort function with the template parameter "Up" set to false:
// radixSort::sort<false>(num, keyArray, payloadArrays...);
//
// For compilation with gcc, compile with the following flags:
// -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi2
// 
// The library requires at least C++17.
//
//=============================================================================


#ifndef _RADIX_SORT_H_
#define _RADIX_SORT_H_

#include <cstdlib>
#include <iostream>
#ifdef __AVX512F__

#include <bitset>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <popcntintrin.h>
#include <tuple>
#include <type_traits>
#include <x86intrin.h>

#define INLINE inline __attribute__((always_inline))
template <typename> inline constexpr bool always_false_v = false;

namespace radixSort {

namespace simd {
// simd library inspired by
// the T-SIMD library written by Prof. Dr.-Ing. Ralf Möller:
// https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html

template <typename T, int Bytes, typename = void> struct _MMRegType {
  static_assert(always_false_v<T>, "Unsupported type or number of bytes");
};
template <typename T>
struct _MMRegType<T, 64, std::enable_if_t<std::is_integral_v<T>>> {
  using type = __m512i;
};
template <typename T>
struct _MMRegType<T, 32, std::enable_if_t<std::is_integral_v<T>>> {
  using type = __m256i;
};
template <typename T>
struct _MMRegType<T, 16, std::enable_if_t<std::is_integral_v<T>>> {
  using type = __m128i;
};
template <typename T>
struct _MMRegType<T, 8, std::enable_if_t<std::is_integral_v<T>>> {
  using type = __m128i;
};
template <> struct _MMRegType<float, 64> { using type = __m512; };
template <> struct _MMRegType<float, 32> { using type = __m256; };
template <> struct _MMRegType<float, 16> { using type = __m128; };
template <> struct _MMRegType<float, 8> { using type = __m128; };
template <> struct _MMRegType<double, 64> { using type = __m512d; };
template <> struct _MMRegType<double, 32> { using type = __m256d; };
template <> struct _MMRegType<double, 16> { using type = __m128d; };
template <> struct _MMRegType<double, 8> { using type = __m128d; };

template <typename T, int Bytes>
using MMRegType = typename _MMRegType<T, Bytes>::type;

template <int X>
static constexpr bool is_power_of_two = X > 0 && (X & (X - 1)) == 0;

template <typename T, int Bytes = 64, typename = void> struct Vec;

template <typename T, int Bytes>
struct Vec<T, Bytes,
           std::enable_if_t<(Bytes <= 64) && is_power_of_two<Bytes>>> {
  MMRegType<T, Bytes> mmReg;
  static constexpr int numElems = Bytes / sizeof(T);
  Vec() = default;
  Vec(const MMRegType<T, Bytes> x) : mmReg(x) {}
  Vec &operator=(const MMRegType<T, Bytes> x) {
    mmReg = x;
    return *this;
  }
  operator MMRegType<T, Bytes>() const { return mmReg; }
};

template <typename T, int Bytes>
struct Vec<T, Bytes, std::enable_if_t<(Bytes > 64) && is_power_of_two<Bytes>>> {
  MMRegType<T, 64> mmReg[Bytes / 64];
  static constexpr int numElems = Bytes / sizeof(T);
  static constexpr int numRegs = Bytes / 64;
  Vec() = default;
  MMRegType<T, 64> &operator[](int i) { return mmReg[i]; }
  const MMRegType<T, 64> &operator[](int i) const { return mmReg[i]; }
};

template <int Size> struct _MaskType {
  static_assert(always_false_v<_MaskType<Size>>, "Unsupported mask size");
};
template <> struct _MaskType<64> { using type = __mmask64; };
template <> struct _MaskType<32> { using type = __mmask32; };
template <> struct _MaskType<16> { using type = __mmask16; };
template <> struct _MaskType<8> { using type = __mmask8; };

template <int Size> using MaskType = typename _MaskType<Size>::type;

template <int Size> struct Mask {
  MaskType<Size> k;
  Mask() = default;
  Mask(const MaskType<Size> &x) : k(x) {}
  Mask &operator=(const MaskType<Size> &x) {
    k = x;
    return *this;
  }
  operator MaskType<Size>() const { return k; }
};

template <typename Tdst, typename Tsrc, int Bytes>
static INLINE Vec<Tdst, Bytes> reinterpret(const Vec<Tsrc, Bytes> &vec) {
  return reinterpret_cast<const Vec<Tdst, Bytes> &>(vec);
}

template <typename T, int Bytes = 64,
          typename = std::enable_if_t<std::is_integral_v<T>>>
static INLINE Vec<T, Bytes> set1(const T a) {
  if constexpr (Bytes == 64) {
    if constexpr (sizeof(T) == 1) {
      return _mm512_set1_epi8(a);
    } else if constexpr (sizeof(T) == 2) {
      return _mm512_set1_epi16(a);
    } else if constexpr (sizeof(T) == 4) {
      return _mm512_set1_epi32(a);
    } else if constexpr (sizeof(T) == 8) {
      return _mm512_set1_epi64(a);
    } else {
      static_assert(always_false_v<T>, "Unsupported integer size");
    }
  } else if constexpr (Bytes == 32) {
    if constexpr (sizeof(T) == 1) {
      return _mm256_set1_epi8(a);
    } else if constexpr (sizeof(T) == 2) {
      return _mm256_set1_epi16(a);
    } else if constexpr (sizeof(T) == 4) {
      return _mm256_set1_epi32(a);
    } else {
      static_assert(always_false_v<T>, "Unsupported integer size");
    }
  } else if constexpr (Bytes == 16 || Bytes == 8) {
    if constexpr (sizeof(T) == 1) {
      return _mm_set1_epi8(a);
    } else if constexpr (sizeof(T) == 2) {
      return _mm_set1_epi16(a);
    } else if constexpr (sizeof(T) == 4) {
      return _mm_set1_epi32(a);
    } else {
      static_assert(always_false_v<T>, "Unsupported integer size");
    }
  } else {
    static_assert(always_false_v<T>, "Unsupported integer size");
  }
}

template <int Bytes = 64, typename T>
static INLINE Vec<T, Bytes> loadu(const T *p) {
  if constexpr (Bytes == 64) {
    if constexpr (std::is_integral_v<T>) {
      return _mm512_loadu_si512(p);
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm512_loadu_ps(p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm512_loadu_pd(p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 32) {
    if constexpr (std::is_integral_v<T>) {
      return _mm256_loadu_si256((__m256i_u *)p);
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm256_loadu_ps(p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm256_loadu_pd(p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 16) {
    if constexpr (std::is_integral_v<T>) {
      return _mm_loadu_si128((__m128i_u *)p);
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm_loadu_ps(p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm_loadu_pd(p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#if defined(__AVX512BW__) && defined(__AVX512VL__)
  } else if constexpr (Bytes == 8) {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (sizeof(T) == 1) {
        return _mm_maskz_loadu_epi8(0xff, p);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#endif
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Vec<T, Bytes> result;
    for (int i = 0; i < Vec<T, Bytes>::numRegs; i++) {
      if constexpr (std::is_integral_v<T>) {
        result[i] = _mm512_loadu_si512(p + i * Vec<T, 64>::numElems);
      } else if constexpr (std::is_same_v<T, float>) {
        result[i] = _mm512_loadu_ps(p + i * Vec<T, 64>::numElems);
      } else if constexpr (std::is_same_v<T, double>) {
        result[i] = _mm512_loadu_pd(p + i * Vec<T, 64>::numElems);
      } else {
        static_assert(always_false_v<T>, "Unsupported type");
      }
    }
    return result;
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <int Bytes = 64, typename T>
static INLINE Vec<T, Bytes> maskz_loadu(const Mask<Vec<T, Bytes>::numElems> m,
                                        const T *p) {
  if constexpr (Bytes == 64) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm512_maskz_loadu_epi8(m, p);
      } else if constexpr (sizeof(T) == 2) {
        return _mm512_maskz_loadu_epi16(m, p);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm512_maskz_loadu_epi32(m, p);
      } else if constexpr (sizeof(T) == 8) {
        return _mm512_maskz_loadu_epi64(m, p);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm512_maskz_loadu_ps(m, p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm512_maskz_loadu_pd(m, p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm256_maskz_loadu_epi8(m, (__m256i_u *)p);
      } else if constexpr (sizeof(T) == 2) {
        return _mm256_maskz_loadu_epi16(m, (__m256i_u *)p);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm256_maskz_loadu_epi32(m, (__m256i_u *)p);
      } else if constexpr (sizeof(T) == 8) {
        return _mm256_maskz_loadu_epi64(m, (__m256i_u *)p);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm256_maskz_loadu_ps(m, p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm256_maskz_loadu_pd(m, p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 16) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm_maskz_loadu_epi8(m, p);
      } else if constexpr (sizeof(T) == 2) {
        return _mm_maskz_loadu_epi16(m, p);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm_maskz_loadu_epi32(m, p);
      } else if constexpr (sizeof(T) == 8) {
        return _mm_maskz_loadu_epi64(m, p);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm_maskz_loadu_ps(m, p);
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm_maskz_loadu_pd(m, p);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 8) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm_maskz_loadu_epi8(m & 0xff, p);
      } else
#endif
      {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#endif // __AVX512VL__
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Vec<T, Bytes> result;
    Mask<Vec<T, Bytes>::numElems> mask = m;
    for (int i = 0; i < Vec<T, Bytes>::numRegs; i++) {
      if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
        if constexpr (sizeof(T) == 1) {
          result[i] =
              _mm512_maskz_loadu_epi8(mask, p + i * Vec<T, 64>::numElems);
        } else if constexpr (sizeof(T) == 2) {
          result[i] =
              _mm512_maskz_loadu_epi16(mask, p + i * Vec<T, 64>::numElems);
        } else
#endif
            if constexpr (sizeof(T) == 4) {
          result[i] =
              _mm512_maskz_loadu_epi32(mask, p + i * Vec<T, 64>::numElems);
        } else if constexpr (sizeof(T) == 8) {
          result[i] =
              _mm512_maskz_loadu_epi64(mask, p + i * Vec<T, 64>::numElems);
        } else {
          static_assert(always_false_v<T>, "Unsupported integer size");
        }
      } else if constexpr (std::is_same_v<T, float>) {
        result[i] = _mm512_maskz_loadu_ps(mask, p + i * Vec<T, 64>::numElems);
      } else if constexpr (std::is_same_v<T, double>) {
        result[i] = _mm512_maskz_loadu_pd(mask, p + i * Vec<T, 64>::numElems);
      } else {
        static_assert(always_false_v<T>, "Unsupported type");
      }
      mask = mask >> Vec<T, 64>::numElems;
    }
    return result;
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <typename T, int Bytes>
static INLINE void mask_compressstoreu(T *p,
                                       const Mask<Vec<T, Bytes>::numElems> m,
                                       const Vec<T, Bytes> v) {
  if constexpr (Bytes == 64) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512VBMI2__
      if constexpr (sizeof(T) == 1) {
        _mm512_mask_compressstoreu_epi8(p, m, v);
      } else if constexpr (sizeof(T) == 2) {
        _mm512_mask_compressstoreu_epi16(p, m, v);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        _mm512_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) == 8) {
        _mm512_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      _mm512_mask_compressstoreu_ps(p, m, v);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm512_mask_compressstoreu_pd(p, m, v);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512VBMI2__
      if constexpr (sizeof(T) == 1) {
        _mm256_mask_compressstoreu_epi8(p, m, v);
      } else if constexpr (sizeof(T) == 2) {
        _mm256_mask_compressstoreu_epi16(p, m, v);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        _mm256_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) == 8) {
        _mm256_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      _mm256_mask_compressstoreu_ps(p, m, v);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm256_mask_compressstoreu_pd(p, m, v);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 16) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512VBMI2__
      if constexpr (sizeof(T) == 1) {
        _mm_mask_compressstoreu_epi8(p, m, v);
      } else if constexpr (sizeof(T) == 2) {
        _mm_mask_compressstoreu_epi16(p, m, v);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        _mm_mask_compressstoreu_epi32(p, m, v);
      } else if constexpr (sizeof(T) == 8) {
        _mm_mask_compressstoreu_epi64(p, m, v);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      _mm_mask_compressstoreu_ps(p, m, v);
    } else if constexpr (std::is_same_v<T, double>) {
      _mm_mask_compressstoreu_pd(p, m, v);
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 8) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512VBMI2__
      if constexpr (sizeof(T) == 1) {
        _mm_mask_compressstoreu_epi8(p, m & 0xff, v);
      } else
#endif
      {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#endif
  } else if constexpr (Bytes == 128 || Bytes == 256 || Bytes == 512) {
    Mask<Vec<T, Bytes>::numElems> mask = m;
    for (int i = 0; i < Vec<T, Bytes>::numRegs; i++) {
      if constexpr (std::is_integral_v<T>) {
        if constexpr (sizeof(T) == 2) {
          _mm512_mask_compressstoreu_epi16(p, mask, v[i]);
        } else if constexpr (sizeof(T) == 4) {
          _mm512_mask_compressstoreu_epi32(p, mask, v[i]);
        } else if constexpr (sizeof(T) == 8) {
          _mm512_mask_compressstoreu_epi64(p, mask, v[i]);
        } else {
          static_assert(always_false_v<T>, "Unsupported integer size");
        }
      } else if constexpr (std::is_same_v<T, float>) {
        _mm512_mask_compressstoreu_ps(p, mask, v[i]);
      } else if constexpr (std::is_same_v<T, double>) {
        _mm512_mask_compressstoreu_pd(p, mask, v[i]);
      } else {
        static_assert(always_false_v<T>, "Unsupported type");
      }
      p += kpopcnt((Mask<Vec<T, 64>::numElems>)mask);
      mask = mask >> Vec<T, 64>::numElems;
    }
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <typename T, int Bytes>
static INLINE Mask<Vec<T, Bytes>::numElems> test_mask(const Vec<T, Bytes> a,
                                                      const Vec<T, Bytes> b) {
  if constexpr (Bytes == 64) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm512_test_epi8_mask(a, b);
      } else if constexpr (sizeof(T) == 2) {
        return _mm512_test_epi16_mask(a, b);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm512_test_epi32_mask(a, b);
      } else if constexpr (sizeof(T) == 8) {
        return _mm512_test_epi64_mask(a, b);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm512_test_epi32_mask(reinterpret<uint32_t>(a),
                                    reinterpret<uint32_t>(b));
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm512_test_epi64_mask(reinterpret<uint64_t>(a),
                                    reinterpret<uint64_t>(b));
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#ifdef __AVX512VL__
  } else if constexpr (Bytes == 32) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm256_test_epi8_mask(a, b);
      } else if constexpr (sizeof(T) == 2) {
        return _mm256_test_epi16_mask(a, b);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm256_test_epi32_mask(a, b);
      } else if constexpr (sizeof(T) == 8) {
        return _mm256_test_epi64_mask(a, b);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm256_test_epi32_mask(reinterpret<uint32_t>(a),
                                    reinterpret<uint32_t>(b));
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm256_test_epi64_mask(reinterpret<uint64_t>(a),
                                    reinterpret<uint64_t>(b));
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
  } else if constexpr (Bytes == 16 || Bytes == 8) {
    if constexpr (std::is_integral_v<T>) {
#ifdef __AVX512BW__
      if constexpr (sizeof(T) == 1) {
        return _mm_test_epi8_mask(a, b);
      } else if constexpr (sizeof(T) == 2) {
        return _mm_test_epi16_mask(a, b);
      } else
#endif
          if constexpr (sizeof(T) == 4) {
        return _mm_test_epi32_mask(a, b);
      } else if constexpr (sizeof(T) == 8) {
        return _mm_test_epi64_mask(a, b);
      } else {
        static_assert(always_false_v<T>, "Unsupported integer size");
      }
    } else if constexpr (std::is_same_v<T, float>) {
      return _mm_test_epi32_mask(reinterpret<uint32_t>(a),
                                 reinterpret<uint32_t>(b));
    } else if constexpr (std::is_same_v<T, double>) {
      return _mm_test_epi64_mask(reinterpret<uint64_t>(a),
                                 reinterpret<uint64_t>(b));
    } else {
      static_assert(always_false_v<T>, "Unsupported type");
    }
#endif
  } else {
    static_assert(always_false_v<T>, "Unsupported vector size");
  }
}

template <int Size> static INLINE int kpopcnt(const Mask<Size> m) {
  return _mm_popcnt_u64(m);
}

template <int Size> static INLINE Mask<Size> knot(const Mask<Size> m) {
#ifdef __AVX512DQ__
  if constexpr (Size == 8) {
    return _knot_mask8(m);
  } else
#endif
      if constexpr (Size == 16) {
    return _knot_mask16(m);
#ifdef __AVX512BW__
  } else if constexpr (Size == 32) {
    return _knot_mask32(m);
  } else if constexpr (Size == 64) {
    return _knot_mask64(m);
#endif
  } else {
    static_assert(always_false_v<Mask<Size>>, "Unsupported mask size");
  }
}

template <int Size>
static INLINE Mask<Size> kand(const Mask<Size> m1, const Mask<Size> m2) {
#ifdef __AVX512DQ__
  if constexpr (Size == 8) {
    return _kand_mask8(m1, m2);
  } else
#endif
      if constexpr (Size == 16) {
    return _kand_mask16(m1, m2);
#ifdef __AVX512BW__
  } else if constexpr (Size == 32) {
    return _kand_mask32(m1, m2);
  } else if constexpr (Size == 64) {
    return _kand_mask64(m1, m2);
#endif
  } else {
    static_assert(always_false_v<Mask<Size>>, "Unsupported mask size");
  }
}
} // namespace simd

using SortIndex = ssize_t;

template <int Bytes> struct _UInt;
template <> struct _UInt<1> { using type = uint8_t; };
template <> struct _UInt<2> { using type = uint16_t; };
template <> struct _UInt<4> { using type = uint32_t; };
template <> struct _UInt<8> { using type = uint64_t; };

template <int Bytes> using UInt = typename _UInt<Bytes>::type;

template <typename T> INLINE bool isBitSet(int bitNo, T val) {
  UInt<sizeof(T)> valAsUInt;
  memcpy(&valAsUInt, &val, sizeof(T));
  return (valAsUInt >> bitNo) & 1;
}

template <typename K, bool Up, bool IsHighestBit, bool IsRightSide>
constexpr bool bitDirUp() {
  if constexpr (std::__is_unsigned_integer<K>::value) {
    return Up;
  }
  if constexpr (std::__is_signed_integer<K>::value) {
    if constexpr (IsHighestBit) {
      return !Up;
    } else {
      return Up;
    }
  }
  if constexpr (std::is_floating_point<K>::value) {
    if constexpr (IsHighestBit) {
      return !Up;
    } else {
      return IsRightSide;
    }
  }
  return Up;
}

struct CmpSorterInsertionSort {
  static std::string name() { return "CmpSorterInsertionSort"; }

  template <bool Up, typename K, typename... Ps>
  static INLINE void sort(SortIndex left, SortIndex right, K *keys,
                          Ps *...payloads) {
    for (SortIndex i = left + 1; i <= right; i++) {
      K key = keys[i];
      std::tuple<Ps...> payload = std::make_tuple(payloads[i]...);
      SortIndex j = i;
      while (j > left && (Up ? key < keys[j - 1] : key > keys[j - 1])) {
        keys[j] = keys[j - 1];
        ((payloads[j] = payloads[j - 1]), ...);
        j--;
      }
      keys[j] = key;
      std::apply([&](Ps &...p) { ((payloads[j] = p), ...); }, payload);
    }
  }
};

#if defined(SORT512_HPP) && defined(SORT512KV_HPP)
struct CmpSorterBramasSmallSort {
  static std::string name() { return "CmpSorterBramasSmallSort"; }

  template <bool Up, typename K, typename... Ps>
  static INLINE void sort(SortIndex left, SortIndex right, K *keys,
                          Ps *...payloads) {
    static_assert(std::__is_one_of<K, double, int>::value,
                  "BramasSmallSort only supports int and double");
    static_assert(Up, "BramasSmallSort only supports sorting up");

    if constexpr (sizeof...(Ps) == 0) {
      Sort512::SmallSort16V(keys + left, right - left + 1);
    } else if constexpr (sizeof...(Ps) == 1) {
      static_assert(std::is_same_v<K, Ps...>,
                    "key and payload must have the same type");
      Sort512kv::SmallSort16V(keys + left,
                              std::get<0>(std::make_tuple(payloads...)) + left,
                              right - left + 1);
    } else {
      static_assert(always_false_v<K>,
                    "BramasSmallSort only supports one or zero payloads");
    }
  }
};
#endif

struct CmpSorterNoSort {
  static std::string name() { return "CmpSorterNoSort"; }

  template <bool Up, typename K, typename... Ps>
  static INLINE void sort(SortIndex left, SortIndex right, K *keys,
                          Ps *...payloads) {
    // do nothing
  }
};

struct BitSorterSequential {
  static std::string name() { return "BitSorterSequential"; }

  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static INLINE SortIndex sortBit(int bitNo, SortIndex left, SortIndex right,
                                  K *keys, Ps *...payloads) {
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
  static INLINE SortIndex sortBit(int bitNo, SortIndex left, SortIndex right,
                                  K *keys, Ps *...payloads) {
    return (left + right) / 2;
  }
};

#ifdef __AVX512F__
template <bool OneReg = false> struct BitSorterSIMD {
  static std::string name() {
    if constexpr (OneReg) {
      return "BitSorterSIMD<OneReg>";
    } else {
      return "BitSorterSIMD";
    }
  }

  template <typename K, typename... Ps>
  static constexpr SortIndex
      numElemsPerVec = OneReg ? 64 / std::max({sizeof(K), sizeof(Ps)...})
                              : 64 / sizeof(K);

  template <bool Up, bool IsHighestBit, bool IsRightSide, typename K,
            typename... Ps>
  static INLINE SortIndex sortBit(int bitNo, SortIndex left, SortIndex right,
                                  K *keys, Ps *...payloads) {
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
      restMask = (simd::knot(simd::Mask<_numElemsPerVec>(0))) >>
                 (_numElemsPerVec - numElemsRest);
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
  static INLINE std::tuple<simd::Mask<numElemsPerVec<K, Ps...>>,
                           simd::Mask<numElemsPerVec<K, Ps...>>>
  getSortMasks(simd::Vec<K, numElemsPerVec<K, Ps...> * sizeof(K)> keyVec,
               int bitNo) {
    auto bitMaskVec = simd::reinterpret<K>(
        simd::set1<UInt<sizeof(K)>, numElemsPerVec<K, Ps...> * sizeof(K)>(
            UInt<sizeof(K)>(1) << bitNo));
    if constexpr (bitDirUp<K, Up, IsHighestBit, IsRightSide>()) {
      auto sortMaskRight = simd::test_mask(keyVec, bitMaskVec);
      auto sortMaskLeft = simd::knot(sortMaskRight);
      return std::make_tuple(sortMaskLeft, sortMaskRight);
    } else {
      auto sortMaskLeft = simd::test_mask(keyVec, bitMaskVec);
      auto sortMaskRight = simd::knot(sortMaskLeft);
      return std::make_tuple(sortMaskLeft, sortMaskRight);
    }
  }

  template <typename K, typename... Ps>
  static INLINE void compress_store_left_right(
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
#endif

template <bool Up, typename BitSorter, typename CmpSorter,
          bool IsRightSide = false, bool IsHighestBit = true, typename K,
          typename... Ps>
void radixRecursion(int bitNo, SortIndex cmpSortThreshold, SortIndex left,
                    SortIndex right, K *keys, Ps *...payloads) {
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
#endif
  radixRecursion<Up, BitSorter, CmpSorter>(sizeof(K) * 8 - 1, cmpSortThreshold,
                                           0, num - 1, keys, payloads...);
}

template <bool Up = true, typename K, typename... Ps>
void sort(SortIndex num, K *keys, Ps *...payloads) {
  sort<Up, BitSorterSIMD<>, CmpSorterInsertionSort>(16, num, keys, payloads...);
}

} // namespace radixSort

#endif // __AVX512F__

#endif
