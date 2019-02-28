#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;

void main() {
    gl_Position = vec4(vPositions, 1.0);
}
