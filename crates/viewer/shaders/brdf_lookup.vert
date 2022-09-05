#version 450

layout(location = 0) in vec2 vPositions;
layout(location = 1) in vec2 vCoords;

layout(location = 0) out vec2 oCoords;

void main() {
    oCoords = vCoords;
    gl_Position = vec4(vPositions, 0.0, 1.0);
}
