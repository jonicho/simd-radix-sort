# simd-radix-sort

Code for the bachelor's thesis "SIMD Implementation of MSB Radix Sort with Separate Key and Payload Datastreams" at the Bielefeld University.

Requires C++20.

## Header-only library

The file `radixSort.hpp` provides a generic AVX-512 implementation of MSB Radix Sort for sorting datasets with a key and an arbitrary number of payloads as a header-only library.

## Tests

The file `test.cpp` provides a simple test program for the library. It can be compiled with:
    
```bash
$ make test
```

and run with:
    
```bash
$ ./build/test
```

## Experiments for the bachelor's thesis

The file `perf.cpp` provides the program that was used to generate the data for the experiments of the bachelor's thesis.
It can be compiled with:
    
```bash
$ make perf
```

and run with:
    
```bash
$ ./build/perf
```

Note that to be able to run all tests of the `perf.cpp` program, the following libraries need to be installed:
- [Intel IPP library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html), the environment variable `IPPROOT` needs to be set.
- The generic SIMD implementation of MSB Radix Sort by Prof. Dr.-Ing. Ralf Möller (the file `SIMDRadixSortGeneric.H`) included in the [T-SIMD](https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html) library must be present in the `moeller/` directory.

If either of these libraries is not present, the `perf.cpp` program will be able to be compiled and run, but the tests that require either of them will not be executed.
