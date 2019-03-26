/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#pragma once

#include <thrift/lib/cpp2/Thrift.h>
#include <thrift/lib/cpp2/protocol/Protocol.h>
#include <thrift/lib/cpp/TApplicationException.h>
#include <folly/io/IOBuf.h>
#include <folly/io/Cursor.h>
#include <boost/operators.hpp>




namespace thpp {

class ThriftTensor;
class ThriftStorage;

enum class ThriftTensorDataType {
  BYTE = 1,
  INT32 = 2,
  INT64 = 3,
  FLOAT = 4,
  DOUBLE = 5
};

extern const std::map<ThriftTensorDataType, const char*> _ThriftTensorDataType_VALUES_TO_NAMES;
extern const std::map<const char*, ThriftTensorDataType, apache::thrift::ltstr> _ThriftTensorDataType_NAMES_TO_VALUES;

} // thpp
namespace apache { namespace thrift {

template <> const char* TEnumTraitsBase< ::thpp::ThriftTensorDataType>::findName( ::thpp::ThriftTensorDataType value);
template <> bool TEnumTraitsBase< ::thpp::ThriftTensorDataType>::findValue(const char* name,  ::thpp::ThriftTensorDataType* outValue);

template <> constexpr  ::thpp::ThriftTensorDataType TEnumTraits< ::thpp::ThriftTensorDataType>::min() {
  return  ::thpp::ThriftTensorDataType::BYTE;
}

template <> constexpr  ::thpp::ThriftTensorDataType TEnumTraits< ::thpp::ThriftTensorDataType>::max() {
  return  ::thpp::ThriftTensorDataType::DOUBLE;
}

}} // apache::thrift
namespace thpp {

enum class ThriftTensorEndianness {
  LITTLE = 1,
  BIG = 2,
  NATIVE = 3
};

extern const std::map<ThriftTensorEndianness, const char*> _ThriftTensorEndianness_VALUES_TO_NAMES;
extern const std::map<const char*, ThriftTensorEndianness, apache::thrift::ltstr> _ThriftTensorEndianness_NAMES_TO_VALUES;

} // thpp
namespace apache { namespace thrift {

template <> const char* TEnumTraitsBase< ::thpp::ThriftTensorEndianness>::findName( ::thpp::ThriftTensorEndianness value);
template <> bool TEnumTraitsBase< ::thpp::ThriftTensorEndianness>::findValue(const char* name,  ::thpp::ThriftTensorEndianness* outValue);

template <> constexpr  ::thpp::ThriftTensorEndianness TEnumTraits< ::thpp::ThriftTensorEndianness>::min() {
  return  ::thpp::ThriftTensorEndianness::LITTLE;
}

template <> constexpr  ::thpp::ThriftTensorEndianness TEnumTraits< ::thpp::ThriftTensorEndianness>::max() {
  return  ::thpp::ThriftTensorEndianness::NATIVE;
}

}} // apache::thrift
namespace thpp {

typedef folly::IOBuf IOBuf;

class ThriftTensor : private boost::totally_ordered<ThriftTensor> {
 public:

  ThriftTensor() {}
  // FragileConstructor for use in initialization lists only

  ThriftTensor(apache::thrift::FragileConstructor,  ::thpp::ThriftTensorDataType dataType__arg,  ::thpp::ThriftTensorEndianness endianness__arg, std::vector<int64_t> sizes__arg,  ::thpp::IOBuf data__arg) :
      dataType(std::move(dataType__arg)),
      endianness(std::move(endianness__arg)),
      sizes(std::move(sizes__arg)),
      data(std::move(data__arg)) {}

  ThriftTensor(ThriftTensor&&) = default;

  ThriftTensor(const ThriftTensor&) = default;

  ThriftTensor& operator=(ThriftTensor&&) = default;

  ThriftTensor& operator=(const ThriftTensor&) = default;
  void __clear();

  virtual ~ThriftTensor() throw() {}

   ::thpp::ThriftTensorDataType dataType;
   ::thpp::ThriftTensorEndianness endianness;
  std::vector<int64_t> sizes;
   ::thpp::IOBuf data;

  struct __isset {
    __isset() {
      __clear();
    }

    void __clear() {
      data = false;
    }

    bool data;
  } __isset;
  bool operator==(const ThriftTensor& rhs) const;
  bool operator < (const ThriftTensor& rhs) const;

  template <class Protocol_>
  uint32_t read(Protocol_* iprot);
  template <class Protocol_>
  uint32_t serializedSize(Protocol_* prot_) const;
  template <class Protocol_>
  uint32_t serializedSizeZC(Protocol_* prot_) const;
  template <class Protocol_>
  uint32_t write(Protocol_* prot_) const;
};

void swap(ThriftTensor& a, ThriftTensor& b);

} // thpp
namespace apache { namespace thrift {

template <> inline void Cpp2Ops< ::thpp::ThriftTensor>::clear( ::thpp::ThriftTensor* obj) {
  return obj->__clear();
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftTensor>::write(Protocol* proto, const  ::thpp::ThriftTensor* obj) {
  return obj->write(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftTensor>::read(Protocol* proto,   ::thpp::ThriftTensor* obj) {
  return obj->read(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftTensor>::serializedSize(Protocol* proto, const  ::thpp::ThriftTensor* obj) {
  return obj->serializedSize(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftTensor>::serializedSizeZC(Protocol* proto, const  ::thpp::ThriftTensor* obj) {
  return obj->serializedSizeZC(proto);
}

}} // apache::thrift
namespace thpp {

class ThriftStorage : private boost::totally_ordered<ThriftStorage> {
 public:

  ThriftStorage() {}
  // FragileConstructor for use in initialization lists only

  ThriftStorage(apache::thrift::FragileConstructor,  ::thpp::ThriftTensorDataType dataType__arg,  ::thpp::ThriftTensorEndianness endianness__arg,  ::thpp::IOBuf data__arg) :
      dataType(std::move(dataType__arg)),
      endianness(std::move(endianness__arg)),
      data(std::move(data__arg)) {}

  ThriftStorage(ThriftStorage&&) = default;

  ThriftStorage(const ThriftStorage&) = default;

  ThriftStorage& operator=(ThriftStorage&&) = default;

  ThriftStorage& operator=(const ThriftStorage&) = default;
  void __clear();

  virtual ~ThriftStorage() throw() {}

   ::thpp::ThriftTensorDataType dataType;
   ::thpp::ThriftTensorEndianness endianness;
   ::thpp::IOBuf data;

  struct __isset {
    __isset() {
      __clear();
    }

    void __clear() {
      data = false;
    }

    bool data;
  } __isset;
  bool operator==(const ThriftStorage& rhs) const;
  bool operator < (const ThriftStorage& rhs) const;

  template <class Protocol_>
  uint32_t read(Protocol_* iprot);
  template <class Protocol_>
  uint32_t serializedSize(Protocol_* prot_) const;
  template <class Protocol_>
  uint32_t serializedSizeZC(Protocol_* prot_) const;
  template <class Protocol_>
  uint32_t write(Protocol_* prot_) const;
};

void swap(ThriftStorage& a, ThriftStorage& b);

} // thpp
namespace apache { namespace thrift {

template <> inline void Cpp2Ops< ::thpp::ThriftStorage>::clear( ::thpp::ThriftStorage* obj) {
  return obj->__clear();
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftStorage>::write(Protocol* proto, const  ::thpp::ThriftStorage* obj) {
  return obj->write(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftStorage>::read(Protocol* proto,   ::thpp::ThriftStorage* obj) {
  return obj->read(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftStorage>::serializedSize(Protocol* proto, const  ::thpp::ThriftStorage* obj) {
  return obj->serializedSize(proto);
}

template <> template <class Protocol> inline uint32_t Cpp2Ops< ::thpp::ThriftStorage>::serializedSizeZC(Protocol* proto, const  ::thpp::ThriftStorage* obj) {
  return obj->serializedSizeZC(proto);
}

}} // apache::thrift
namespace thpp {

} // thpp