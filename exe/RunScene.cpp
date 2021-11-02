//
// Created by jbs on 21. 10. 31..
//

#include <SceneInterpreter.h>
using namespace iswy;
int main() {

    SceneInterpreter sceneInterpreter;
    thread camThread(&SceneInterpreter::cameraThread,&sceneInterpreter);
    thread visThread(&SceneInterpreter::visThread,&sceneInterpreter);
    camThread.join();
    visThread.join();

    return 0;
}