#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 oCoords;

layout(binding = 0) uniform sampler2D normalsSampler;
layout(binding = 1) uniform sampler2D depthSampler;

layout(location = 0) out vec4 finalColor;

void main() {
    finalColor = vec4(1.0);
}
