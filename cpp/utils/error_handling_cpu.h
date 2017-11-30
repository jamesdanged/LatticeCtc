#ifndef TFKERNELS_ERROR_HANDLING_CPU_H
#define TFKERNELS_ERROR_HANDLING_CPU_H

#ifdef USING_NVCC

// #define EXIT_IF_CPU
#define DO_CPU_ONLY(expr)

#else


#include <cstdlib>
// #define EXIT_IF_CPU std::exit(-1)

#define DO_CPU_ONLY(expr) (expr);

#include <string>
#include <exception>
#include <iostream>
#include <sstream>

#define assert_throw(expr) if (!(expr)) throw(lng::ExceptionWithStack(std::string("Assert failed: ") + #expr));
#define assert_abort(expr) if (!(expr)) abort();
#define assert_abort_with_message(expr) \
  if (!(expr)) { \
    std::cerr << "Assertion failed: " << #expr << std::endl << lng::get_backtrace() << std::endl; \
    abort(); \
  }

namespace lng {

  using std::string;
  using std::stringstream;
  using std::endl;

  string get_backtrace();

  class ExceptionWithStack : public std::exception {
  public:
    string what_msg;

    ExceptionWithStack(const char * msg) {
      stringstream ss;
      ss << string(msg) << endl << get_backtrace() << endl;
      what_msg = ss.str();
    }
    ExceptionWithStack(const std::string msg) {
      stringstream ss;
      ss << msg << endl << get_backtrace() << endl;
      what_msg = ss.str();
    }

    virtual const char * what() const _GLIBCXX_USE_NOEXCEPT;
  };



}

#endif



#endif //TFKERNELS_ERROR_HANDLING_CPU_H
