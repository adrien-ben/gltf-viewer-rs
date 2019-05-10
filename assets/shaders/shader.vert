#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;
layout(location = 1) in vec3 vNormals;
layout(location = 2) in vec2 vTexcoords;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(push_constant) uniform Transform {
    mat4 matrix;
} transform;

layout(location = 0) out vec3 oNormals;
layout(location = 1) out vec2 oTexcoords;
layout(location = 2) out vec3 oPositions;

void main() {
    oNormals = (transform.matrix * vec4(vNormals, 0.0)).xyz;
    oTexcoords = vTexcoords;
    oPositions = (transform.matrix * vec4(vPositions, 1.0)).xyz;
    gl_Position = cameraUBO.proj * cameraUBO.view * transform.matrix * vec4(vPositions, 1.0);
}
