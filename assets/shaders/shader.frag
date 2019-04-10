#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords;

layout(push_constant) uniform Material {
    layout(offset = 64) vec3 color;
    layout(offset = 76) int textureId;
} material;

layout(binding = 1) uniform sampler2D texSamplers[64];

layout(location = 0) out vec4 outColor;

const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const vec3 LIGHT_DIR = vec3(1.0, -1.0, -1.0);
const vec3 AMBIENT_COLOR = vec3(0.5, 0.5, 0.5);

vec3 getBaseColor() {
    if(material.textureId == -1) {
        return material.color;
    }
    return pow(texture(texSamplers[material.textureId], oTexcoords).xyz, vec3(2.2)) * material.color;
}

void main() {
    vec3 baseColor = getBaseColor();
    vec3 color = max(dot(normalize(oNormals), -normalize(LIGHT_DIR)), 0.0) * baseColor + AMBIENT_COLOR * baseColor;
    outColor = vec4(color, 1.0);
}
