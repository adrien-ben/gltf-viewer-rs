#version 450
#extension GL_ARB_separate_shader_objects : enable

// -- Inputs --
layout(location = 0) in vec3 oViewSpaceNormal;

// Output
layout(location = 0) out vec4 outNormals;

void main() {
    outNormals = vec4((normalize(oViewSpaceNormal) * 0.5) + 0.5, 0.0);
}
