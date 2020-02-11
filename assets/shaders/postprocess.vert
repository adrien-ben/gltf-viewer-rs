#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vCoords;

layout(location = 0) out vec2 oCoords;

void main() {
    oCoords = vCoords;
    gl_Position = vec4(vPos.x, vPos.y, 0.0, 1.0);
}
