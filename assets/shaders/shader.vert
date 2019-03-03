#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
} cameraUBO;

void main() {
    gl_Position = cameraUBO.proj * cameraUBO.view * vec4(vPositions, 1.0);
}
