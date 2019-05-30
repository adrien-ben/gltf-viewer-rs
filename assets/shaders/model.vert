#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;
layout(location = 1) in vec3 vNormals;
layout(location = 2) in vec2 vTexcoords;
layout(location = 3) in vec4 vTangents;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(binding = 1) uniform TransformUBO {
    mat4 matrix;
} transform;

layout(location = 0) out vec3 oNormals;
layout(location = 1) out vec2 oTexcoords;
layout(location = 2) out vec3 oPositions;
layout(location = 3) out mat3 oTBN;

void main() {
    vec3 normal = normalize((transform.matrix * vec4(vNormals, 0.0)).xyz);
    vec3 tangent = normalize((transform.matrix * vec4(vTangents.xyz, 0.0)).xyz);
    tangent = normalize(tangent - dot(tangent, normal)*normal);
    vec3 bitangent = cross(normal, tangent) * vTangents.w;

    oNormals = normal;
    oTexcoords = vTexcoords;
    oPositions = (transform.matrix * vec4(vPositions, 1.0)).xyz;
    oTBN = mat3(tangent, bitangent, normal);
    gl_Position = cameraUBO.proj * cameraUBO.view * transform.matrix * vec4(vPositions, 1.0);
}
