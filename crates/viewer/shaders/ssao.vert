#version 450
#extension GL_GOOGLE_include_directive : require

#include "libs/camera.glsl"

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vCoords;

layout(binding = 4, set = 2) uniform Frame {
    Camera camera;
};

layout(location = 0) out vec2 oCoords;
layout(location = 1) out vec3 oViewRay;

void main() {
    oCoords = vCoords;
    oViewRay = (camera.invertedProj * vec4(vPos.x, vPos.y, 0.0, 1.0)).xyz;
    gl_Position = vec4(vPos.x, vPos.y, 0.0, 1.0);
}
