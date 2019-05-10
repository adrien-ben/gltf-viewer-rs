#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords;
layout(location = 2) in vec3 oPositions;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(push_constant) uniform Material {
    layout(offset = 64) vec3 color;
    layout(offset = 76) float metallic;
    layout(offset = 80) float roughness;
    layout(offset = 84) int colorTextureId;
    layout(offset = 88) int metallicRoughnessTextureId;
} material;

layout(binding = 1) uniform sampler2D texSamplers[64];

layout(location = 0) out vec4 outColor;

const vec3 LIGHT_DIR = vec3(1.0, -1.0, 0.0);
const vec3 DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;

vec3 getBaseColor() {
    if(material.colorTextureId == -1) {
        return material.color;
    }
    return pow(texture(texSamplers[material.colorTextureId], oTexcoords).rgb, vec3(2.2)) * material.color;
}

float getMetallic() {
    if(material.metallicRoughnessTextureId == -1) {
        return material.metallic;
    }
    return texture(texSamplers[material.metallicRoughnessTextureId], oTexcoords).b * material.metallic;
}

float getRoughness() {
    if(material.metallicRoughnessTextureId == -1) {
        return material.roughness;
    }
    return texture(texSamplers[material.metallicRoughnessTextureId], oTexcoords).g * material.roughness;
}

vec3 f(vec3 f0, vec3 v, vec3 h) {
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
}

float vis(vec3 n, vec3 l, vec3 v, float a) {
    float aa = a * a;
    float nl = max(dot(n, l), 0.0);
    float nv = max(dot(n, v), 0.0);

    return 0.5 / ((nl * sqrt(nv * nv * (1 - aa) + aa)) + (nv * sqrt(nl * nl * (1 - aa) + aa)));
}

float d(float a, vec3 n, vec3 h) {
    float aa = a * a;
    float nh = max(dot(n, h), 0.0);
    float denom = nh * nh * (aa - 1) + 1;

    return aa / (PI * denom * denom);
}



void main() {
    vec3 baseColor = getBaseColor();
    float metallic = getMetallic();
    float roughness = getRoughness();

    vec3 cDiffuse = mix(baseColor * (1.0 - DIELECTRIC_SPECULAR.r), BLACK, metallic);
    vec3 f0 = mix(DIELECTRIC_SPECULAR, baseColor, metallic);
    float a = roughness * roughness;

    vec3 v = normalize(cameraUBO.eye - oPositions);
    vec3 l = -normalize(LIGHT_DIR);
    vec3 n = normalize(oNormals);
    vec3 h = normalize(l + v);

    vec3 f = f(f0, v, h);
    float vis = vis(n, l, v, a);
    float d = d(a, n, h);

    vec3 diffuse = cDiffuse / PI;
    vec3 fDiffuse = (1 - f) * diffuse;
    vec3 fSpecular = f * vis * d;
    vec3 color = fDiffuse + fSpecular;

    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
