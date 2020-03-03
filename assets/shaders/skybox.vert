#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     mat4 invertedProj;
     vec3 eye;
} cameraUBO;

layout(location = 0) out vec3 oPositions;

mat4 getViewAtOrigin() {
    mat4 view = mat4(cameraUBO.view);
    view[3][0] = 0;
    view[3][1] = 0;
    view[3][2] = 0;
    return view;
}

void main() {
    oPositions = vPositions;

    mat4 view = getViewAtOrigin();

    gl_Position = cameraUBO.proj * view * vec4(vPositions, 1.0);
}
