#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;
layout(location = 1) in vec3 vNormals;
layout(location = 2) in vec2 vTexcoords;
layout(location = 3) in vec4 vTangents;
layout(location = 4) in vec4 vWeights;
layout(location = 5) in uvec4 vJoints;
layout(location = 6) in vec4 vColors;

layout(binding = 0, set = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(binding = 2, set = 0) uniform TransformUBO {
    mat4 matrix;
} transform;

layout(binding = 3, set = 0) uniform SkinUBO {
    mat4 jointMatrices[128];
} skin;

layout(location = 0) out vec3 oNormals;
layout(location = 1) out vec2 oTexcoords;
layout(location = 2) out vec3 oPositions;
layout(location = 3) out vec4 oColors;
layout(location = 4) out mat3 oTBN;

void main() {
    mat4 world = transform.matrix;
    if (vWeights != vec4(0.0)) {
        world *= vWeights.x * skin.jointMatrices[vJoints.x]
            + vWeights.y * skin.jointMatrices[vJoints.y]
            + vWeights.z * skin.jointMatrices[vJoints.z]
            + vWeights.w * skin.jointMatrices[vJoints.w];
    }

    vec3 normal = normalize((world * vec4(vNormals, 0.0)).xyz);
    vec3 tangent = normalize((world * vec4(vTangents.xyz, 0.0)).xyz);
    tangent = normalize(tangent - dot(tangent, normal)*normal);
    vec3 bitangent = cross(normal, tangent) * vTangents.w;

    oNormals = normal;
    oTexcoords = vTexcoords;
    oPositions = (world * vec4(vPositions, 1.0)).xyz;
    oTBN = mat3(tangent, bitangent, normal);
    oColors = vColors;
    gl_Position = cameraUBO.proj * cameraUBO.view * world * vec4(vPositions, 1.0);
}
