//
// Created by richard on 10/21/24.
//

#ifndef CMDLINE_H
#define CMDLINE_H

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cmdline {
namespace detail {
class OptionBase {
public:
    virtual bool HasValue() const = 0;
    virtual bool Set() = 0;
    virtual bool Set(const std::string& value) = 0;
    virtual bool HasSet() const = 0;
    virtual bool IsValid() const = 0;
    virtual bool Must() const = 0;
    virtual const std::string& GetName() const = 0;
    virtual char GetShortName() const = 0;
    virtual const std::string& GetDescription() const = 0;
    virtual const std::string& GetShortDescription() const = 0;

    virtual ~OptionBase() {}
};

class OptionWithoutValue : public OptionBase {
    //
};

template<typename T>
class OptionWithValue : public OptionBase {
    //
};

}// namespace detail
}// namespace cmdline

#endif//CMDLINE_H
