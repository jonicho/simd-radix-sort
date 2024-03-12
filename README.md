# simd-radix-sort

A generic and efficient SIMD implementation of MSB Radix Sort with separate key and payload datastreams that supports arbitrary key and payload data types written in C++ accompanied by a bachelor's thesis.

This library is the result of the [bachelor's thesis "SIMD Implementation of MSB Radix Sort with Separate Key and Payload Datastreams" at the Bielefeld University](BachelorsThesis_JonasKeller.pdf).

Requires at least C++20.

## Single header file

The file [ `radixSort.hpp` ](radixSort.hpp) provides a single header version of the radix sort library that has been stripped of classes and functions that were only needed for the benchmarking and testing.

For compilation with gcc, compile with the following flags:

```
-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512vbmi2
```

### Usage

To use the single header version of the library, simply include the file [ `radixSort.hpp` ](radixSort.hpp) in your project.

To sort a dataset, call the sort function like the following:

```cpp
simd_sort::radix_sort::sort(num, keyArray, payloadArrays...); 
```

where `num` is the number of elements in the dataset, `keyArray` is a pointer
to the key array, and `payloadArrays` are the pointers to the payload arrays.

Or, if key and payload datastreams are combined into a single array, call
the sort function like the following:

```cpp
simd_sort::radix_sort::sort(num, (simd_sort::DataElement<K, Ps...> *)combinedArray); 
```

where `num` is the number of elements in the dataset, `combinedArray` is a
pointer to the combined array and `K` and `Ps...` are the types of the key and
the payloads.

By default the sort function sorts in ascending order. To sort in descending
order, call the sort function with the template parameter `Up` set to false:

    

```cpp
simd_sort::radix_sort::sort<false>(num, keyArray, payloadArrays...); 
```

or

```cpp
simd_sort::radix_sort::sort<false>(num, (simd_sort::DataElement<K, Ps...> *) combinedArray); 
```

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

The file [ `perf-thesis.cpp` ](src/perf-thesis.cpp) provides the program that was used to generate the data for the experiments of the bachelor's thesis.
It can be compiled with:

```bash
$ make perf-thesis
```

and run with:

```bash
$ ./build/perf-thesis
```

Note that to be able to run all tests of the [ `perf-thesis.cpp` ](src/perf-thesis.cpp) program, the following libraries need to be installed:
* [Intel IPP library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html), the environment variable `IPPROOT` needs to be set.
* The generic SIMD implementation of MSB Radix Sort by Prof. Dr.-Ing. Ralf Möller (the file `SIMDRadixSortGeneric.H`) included in the [T-SIMD](https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html) library must be present in the [`moeller/`](moeller/) directory.

If either of these libraries is not present, the [ `perf-thesis.cpp` ](src/perf-thesis.cpp) program will still be able to be compiled and run, but the tests that require them will not be executed.

## Quicksort and research paper

There have been works to publish the findings of the bachelor's thesis in a research paper. As part of this, the radix sort implementation was modified to turn it into a quicksort implementation.

However, the quicksort implementation turned out to have a very similar performance to the radix sort implementation. Additionally, there are already efficient and sufficiently generic SIMD sorting algorithms (for example [x86-simd-sort](https://github.com/intel/x86-simd-sort) by Intel) so it was decided that there weren't enough new findings to justify a research paper.

The results of this work can however still be found in the file [ `quick_sort.hpp` ](src/quick_sort.hpp) and the corresponding benchmarks in the file [ `perf-paper.cpp` ](src/perf-paper.cpp).
