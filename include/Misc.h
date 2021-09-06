//
// Created by jbs on 21. 9. 6..
//

#ifndef ZED_OPEN3D_MISC_H
#define ZED_OPEN3D_MISC_H

#include <chrono>
using namespace std::chrono;
namespace misc {
    class Timer {
        steady_clock::time_point t0;
        double measuredTime;
    public:
        Timer() { t0 = steady_clock::now(); }

        double stop(bool isMillisecond = true) {
            if (isMillisecond)
                measuredTime = duration_cast<milliseconds>(steady_clock::now() - t0).count();
            else
                measuredTime = duration_cast<microseconds>(steady_clock::now() - t0).count();
            return measuredTime;
        }
    };

}
#endif //ZED_OPEN3D_MISC_H
