#version 450
#extension GL_GOOGLE_include_directive : require

#include "libs/camera.glsl"

layout(location = 0) in vec3 vPositions;

layout(binding = 0) uniform Frame {
    Camera camera;
};

layout(location = 0) out vec3 oPositions;

mat4 getViewAtOrigin() {
    mat4 view = mat4(camera.view);
    view[3][0] = 0;
    view[3][1] = 0;
    view[3][2] = 0;
    return view;
}

void main() {
    oPositions = vPositions;

    mat4 view = getViewAtOrigin();

    gl_Position = camera.proj * view * vec4(vPositions, 1.0);
}
