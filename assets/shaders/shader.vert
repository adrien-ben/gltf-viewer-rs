#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;
layout(location = 1) in vec3 vNormals;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
} cameraUBO;

layout(push_constant) uniform Transform {
    mat4 matrix;
} transform;

layout(location = 0) out vec3 oNormals;

void main() {
    oNormals = (transform.matrix * vec4(vNormals, 0.0)).xyz;
    gl_Position = cameraUBO.proj * cameraUBO.view * transform.matrix * vec4(vPositions, 1.0);
}
