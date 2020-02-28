#version 450
#extension GL_ARB_separate_shader_objects : enable

// -- Inputs --
layout(location = 0) in vec3 oNormals;

// Output
layout(location = 0) out vec4 outNormals;

void main() {
    outNormals = vec4(oNormals, 0.0);
}
