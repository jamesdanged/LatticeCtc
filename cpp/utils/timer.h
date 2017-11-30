#ifndef TFKERNELS_TIMER_H
#define TFKERNELS_TIMER_H

#include <chrono>
#include <iostream>

// Can wrap this macro around method calls.
// Must declare variables before any usage:
//  Time start_time, stop_time;
#define run_timer_util(CODE) start_time = now(); CODE; stop_time = now(); std::cout << Duration(stop_time - start_time).count() << "s" << std::endl;

namespace lng {

  using Time = std::chrono::time_point<std::chrono::system_clock>;
  const auto now = std::chrono::system_clock::now;
  using Duration = std::chrono::duration<double>;

  class SimpleTimer {
  public:
    Time t_start;
    Time t_last;

    SimpleTimer(): t_start(now()), t_last(t_start) {}

    /**
     * Time in seconds since start (construction).
     */
    inline double since_start() {
      t_last = now();
      return Duration(t_last - t_start).count();
    }

    /**
     * Time in seconds since last call.
     */
    inline double since_last() {
      auto t_curr = now();

      double elapsed = Duration(t_curr - t_last).count();
      t_last = t_curr;
      return elapsed;
    }
  };


}
#endif //TFKERNELS_TIMER_H
