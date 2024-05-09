#version 450
#extension GL_GOOGLE_include_directive : require

#include "libs/camera.glsl"

layout(location = 0) in vec3 vPositions;
layout(location = 1) in vec3 vNormals;
layout(location = 2) in vec2 vTexcoords0;
layout(location = 3) in vec2 vTexcoords1;
layout(location = 4) in vec4 vTangents;
layout(location = 5) in vec4 vWeights;
layout(location = 6) in uvec4 vJoints;
layout(location = 7) in vec4 vColors;

layout(binding = 0, set = 0) uniform Frame {
    Camera camera;
};

layout(binding = 1, set = 0) uniform TransformUBO {
    mat4 matrix;
} transform;

layout(binding = 2, set = 0) uniform SkinUBO {
    mat4 jointMatrices[512];
} skin;

layout(location = 0) out vec3 oViewSpaceNormal;
layout(location = 1) out vec2 oTexcoords0;
layout(location = 2) out vec2 oTexcoords1;
layout(location = 3) out float oAlpha;

void main() {
    mat4 world = transform.matrix;
    if (vWeights != vec4(0.0)) {
        world *= vWeights.x * skin.jointMatrices[vJoints.x]
            + vWeights.y * skin.jointMatrices[vJoints.y]
            + vWeights.z * skin.jointMatrices[vJoints.z]
            + vWeights.w * skin.jointMatrices[vJoints.w];
    }

    oViewSpaceNormal = normalize((camera.view * world * vec4(vNormals, 0.0)).xyz);
    oTexcoords0 = vTexcoords0;
    oTexcoords1 = vTexcoords1;
    oAlpha = vColors.a;

    gl_Position = camera.proj * camera.view * world * vec4(vPositions, 1.0);
}
