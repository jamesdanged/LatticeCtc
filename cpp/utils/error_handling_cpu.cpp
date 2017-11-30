#include "error_handling_cpu.h"

#include <execinfo.h>

namespace lng {
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::string;


  string get_backtrace() {
    void * array[50]; // up to 50 stack frames for now...
    int size = backtrace(array, 50);

    std::stringstream ss;

    ss << "Backtrace returned " << size << " frames\n\n";
    char ** messages = backtrace_symbols(array, size);

    for (int i = 0; i < size && messages != NULL; i++) {
      ss << "[bt]: (" << i << ") " << messages[i] << endl;
    }
    ss << endl;


    free(messages);
    return ss.str();
  }

  const char * ExceptionWithStack::what() const _GLIBCXX_USE_NOEXCEPT
  {
    return what_msg.c_str();
  }
}
