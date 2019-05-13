#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords;
layout(location = 2) in vec3 oPositions;
layout(location = 3) in mat3 oTBN;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(push_constant) uniform Material {
    layout(offset = 64) vec4 colorAndMetallic;
    layout(offset = 80) vec4 emissiveAndRoughness;
    layout(offset = 96) float occlusion;
    layout(offset = 100) int colorTextureId;
    layout(offset = 104) int metallicRoughnessTextureId;
    layout(offset = 108) int emissiveTextureId;
    layout(offset = 112) int normalTextureId;
    layout(offset = 116) int occlusionTextureId;
} material;

layout(binding = 1) uniform sampler2D texSamplers[64];

layout(location = 0) out vec4 outColor;

const vec3 AMBIENT_COLOR = vec3(0.03);
const vec3 LIGHT_DIR = vec3(-1.0, 0.0, 0.0);
const vec3 DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;

vec3 getBaseColor() {
    vec3 color = material.colorAndMetallic.rgb;
    if(material.colorTextureId != -1) {
        color *= pow(texture(texSamplers[material.colorTextureId], oTexcoords).rgb, vec3(2.2));
    }
    return color;
}

float getMetallic() {
    float metallic = material.colorAndMetallic.a;
    if(material.metallicRoughnessTextureId != -1) {
        metallic *= texture(texSamplers[material.metallicRoughnessTextureId], oTexcoords).b;
    }
    return metallic;
}

float getRoughness() {
    float roughness = material.emissiveAndRoughness.a;
    if(material.metallicRoughnessTextureId != -1) {
        roughness *= texture(texSamplers[material.metallicRoughnessTextureId], oTexcoords).g;
    }
    return roughness;
}

vec3 getEmissiveColor() {
    vec3 emissive = material.emissiveAndRoughness.rgb;
    if(material.emissiveTextureId != -1) {
        emissive *= pow(texture(texSamplers[material.emissiveTextureId], oTexcoords).rgb, vec3(2.2));
    }
    return emissive;
}

vec3 getNormal() {
    if (material.normalTextureId != -1) {
        vec3 normal = texture(texSamplers[material.normalTextureId], oTexcoords).rgb * 2.0 - 1.0;
        return normalize(oTBN * normal);
    }
    return normalize(oNormals);
}

vec3 getAmbientColor(vec3 baseColor) {
    float sampledOcclusion = 0.0;
    if (material.occlusionTextureId != -1) {
        sampledOcclusion = texture(texSamplers[material.occlusionTextureId], oTexcoords).r;
    }
    vec3 ambientColor = AMBIENT_COLOR * baseColor;
    return mix(ambientColor, ambientColor * sampledOcclusion, material.occlusion);
}

vec3 f(vec3 f0, vec3 v, vec3 h) {
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
}

float vis(vec3 n, vec3 l, vec3 v, float a) {
    float aa = a * a;
    float nl = max(dot(n, l), 0.0);
    float nv = max(dot(n, v), 0.0);
    float denom = ((nl * sqrt(nv * nv * (1 - aa) + aa)) + (nv * sqrt(nl * nl * (1 - aa) + aa))); 

    if (denom < 0.0) {
        return 0.0;
    }
    return 0.5 / denom;
}

float d(float a, vec3 n, vec3 h) {
    float aa = a * a;
    float nh = max(dot(n, h), 0.0);
    float denom = nh * nh * (aa - 1) + 1;

    return aa / (PI * denom * denom);
}

vec3 computeColor(vec3 baseColor, float metallic, float roughness, vec3 n, vec3 l, vec3 v, vec3 h) {
    vec3 color = vec3(0.0);
    if (dot(n, l) > 0.0) {
        vec3 cDiffuse = mix(baseColor * (1.0 - DIELECTRIC_SPECULAR.r), BLACK, metallic);
        vec3 f0 = mix(DIELECTRIC_SPECULAR, baseColor, metallic);
        float a = roughness * roughness;

        vec3 f = f(f0, v, h);
        float vis = vis(n, l, v, a);
        float d = d(a, n, h);

        vec3 diffuse = cDiffuse / PI;
        vec3 fDiffuse = (1 - f) * diffuse;
        vec3 fSpecular = f * vis * d;
        color = fDiffuse + fSpecular;
    }
    return color;
}

void main() {
    vec3 baseColor = getBaseColor();
    float metallic = getMetallic();
    float roughness = getRoughness();
    vec3 emissive = getEmissiveColor();

    vec3 n = getNormal();
    vec3 l = -normalize(LIGHT_DIR);
    vec3 v = normalize(cameraUBO.eye - oPositions);
    vec3 h = normalize(l + v);

    vec3 ambientColor = getAmbientColor(baseColor);
    vec3 color = ambientColor + computeColor(baseColor, metallic, roughness, n, l, v, h) + emissive;

    color = color/(color + 1.0);
    color = pow(color, vec3(1.0/2.2));
    outColor = vec4(color, 1.0);
}
