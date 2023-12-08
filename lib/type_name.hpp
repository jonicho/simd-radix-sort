#ifndef _TYPE_NAME_H_
#define _TYPE_NAME_H_

#include <cstdint>
#include <typeinfo>

template <typename T>
struct _type_name {
  static constexpr const char *value = typeid(T).name();
};
template <>
struct _type_name<bool> {
  static constexpr const char *value = "bool";
};
template <>
struct _type_name<uint8_t> {
  static constexpr const char *value = "uint8";
};
template <>
struct _type_name<uint16_t> {
  static constexpr const char *value = "uint16";
};
template <>
struct _type_name<uint32_t> {
  static constexpr const char *value = "uint32";
};
template <>
struct _type_name<uint64_t> {
  static constexpr const char *value = "uint64";
};
template <>
struct _type_name<int8_t> {
  static constexpr const char *value = "int8";
};
template <>
struct _type_name<int16_t> {
  static constexpr const char *value = "int16";
};
template <>
struct _type_name<int32_t> {
  static constexpr const char *value = "int32";
};
template <>
struct _type_name<int64_t> {
  static constexpr const char *value = "int64";
};
template <>
struct _type_name<float> {
  static constexpr const char *value = "float";
};
template <>
struct _type_name<double> {
  static constexpr const char *value = "double";
};

template <typename T>
constexpr const char *type_name = _type_name<T>::value;

#endif  // _TYPE_NAME_H_
