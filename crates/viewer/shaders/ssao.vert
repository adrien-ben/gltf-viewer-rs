#version 450

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vCoords;

layout(binding = 4, set = 2) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 invertedProj;
    vec4 eye;
    float zNear;
    float zFar;
} cameraUBO;

layout(location = 0) out vec2 oCoords;
layout(location = 1) out vec3 oViewRay;

void main() {
    oCoords = vCoords;
    oViewRay = (cameraUBO.invertedProj * vec4(vPos.x, vPos.y, 0.0, 1.0)).xyz;
    gl_Position = vec4(vPos.x, vPos.y, 0.0, 1.0);
}
