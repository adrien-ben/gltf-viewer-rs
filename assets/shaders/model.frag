#version 450
#extension GL_ARB_separate_shader_objects : enable

// #define DEBUG_COLOR 1
// #define DEBUG_EMISSIVE 2
// #define DEBUG_METALLIC 3
// #define DEBUG_ROUGHNESS 4
// #define DEBUG_OCCLUSION 5
// #define DEBUG_NORMAL 6
// #define DEBUG_ALPHA 7
// # define DEBUG_UVS 8

// -- Constants --
layout(constant_id = 0) const uint LIGHT_COUNT = 1;

const vec3 DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;

const uint NO_TEXTURE_ID = 255;

const uint ALPHA_MODE_MASK = 1;
const uint ALPHA_MODE_BLEND = 2;

const float MAX_REFLECTION_LOD = 9.0; // last mip mips for 512 px res TODO: specializations ?

const uint DIRECTIONAL_LIGHT_TYPE = 0;
const uint POINT_LIGHT_TYPE = 1;
const uint SPOT_LIGHT_TYPE = 2;

const uint UNLIT_FLAG_UNLIT = 1;

// -- Structures --
struct TextureChannels {
    uint color;
    uint metallicRoughness;
    uint emissive;
    uint normal;
    uint occlusion;
};

struct Light {
    vec4 position;
    vec4 direction;
    vec4 color;
    float intensity;
    float range;
    float angleScale;
    float angleOffset;
    uint type;
};

// -- Inputs --
layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords0;
layout(location = 2) in vec2 oTexcoords1;
layout(location = 3) in vec3 oPositions;
layout(location = 4) in vec4 oColors;
layout(location = 5) in mat3 oTBN;

// -- Push constants
layout(push_constant) uniform Material {
    vec4 color;
    vec4 emissiveAndRoughness;
    float metallic;
    float occlusion;
    // Contains the texture channels for color metallic/roughness emissive and normal
    // [0-7] Color texture channel
    // [8-15] metallic/roughness texture channel
    // [16-23] emissive texture channel
    // [24-31] normals texture channel
    uint colorMetallicRoughnessEmissiveNormalTextureChannels;
    // Contains occlusion texture channel, alpha mode and unlit flag
    // [0-7] Occlusion texture channel
    // [8-15] Alpha mode
    // [16-23] Unlit flag
    uint occlusionTextureChannelAlphaModeAndUnlitFlag;
    float alphaCutoff;
} material;

// -- Descriptors --
layout(binding = 0, set = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    vec3 eye;    
} cameraUBO;
layout(binding = 1, set = 0) uniform Lights {
    Light lights[LIGHT_COUNT];
} lights;
layout(binding = 4, set = 1) uniform samplerCube irradianceMapSampler;
layout(binding = 5, set = 1) uniform samplerCube preFilteredSampler;
layout(binding = 6, set = 1) uniform sampler2D brdfLookupSampler;
layout(binding = 7, set = 2) uniform sampler2D colorSampler;
layout(binding = 8, set = 2) uniform sampler2D normalsSampler;
layout(binding = 9, set = 2) uniform sampler2D metallicRoughnessSampler;
layout(binding = 10, set = 2) uniform sampler2D occlusionSampler;
layout(binding = 11, set = 2) uniform sampler2D emissiveSampler;

// Output
layout(location = 0) out vec4 outColor;

TextureChannels getTextureChannels() {
    return TextureChannels(
        (material.colorMetallicRoughnessEmissiveNormalTextureChannels >> 24) & 255,
        (material.colorMetallicRoughnessEmissiveNormalTextureChannels >> 16) & 255,
        (material.colorMetallicRoughnessEmissiveNormalTextureChannels >> 8) & 255,
        material.colorMetallicRoughnessEmissiveNormalTextureChannels & 255,
        (material.occlusionTextureChannelAlphaModeAndUnlitFlag >> 24) & 255
    );
}

vec2 getUV(uint texChannel) {
    if (texChannel == 0) {
        return oTexcoords0;
    }
    return oTexcoords1;
}

vec4 getBaseColor(TextureChannels textureChannels) {
    vec4 color = material.color;
    if(textureChannels.color != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.color);
        vec4 sampledColor= texture(colorSampler, uv);
        color *= vec4(pow(sampledColor.rgb, vec3(2.2)), sampledColor.a);
    }
    return color * oColors;
}

float getMetallic(TextureChannels textureChannels) {
    float metallic = material.metallic;
    if(textureChannels.metallicRoughness != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.metallicRoughness);
        metallic *= texture(metallicRoughnessSampler, uv).b;
    }
    return metallic;
}

float getRoughness(TextureChannels textureChannels) {
    float roughness = material.emissiveAndRoughness.a;
    if(textureChannels.metallicRoughness != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.metallicRoughness);
        roughness *= texture(metallicRoughnessSampler, uv).g;
    }
    return roughness;
}

vec3 getEmissiveColor(TextureChannels textureChannels) {
    vec3 emissive = material.emissiveAndRoughness.rgb;
    if(textureChannels.emissive != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.emissive);
        emissive *= pow(texture(emissiveSampler, uv).rgb, vec3(2.2));
    }
    return emissive;
}

vec3 getNormal(TextureChannels textureChannels) {
    vec3 normal = normalize(oNormals);
    if (textureChannels.normal != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.normal);
        vec3 normalMap = texture(normalsSampler, uv).rgb * 2.0 - 1.0;
        normal = normalize(oTBN * normalMap);
    }
    
    if (!gl_FrontFacing) {
        normal *= -1.0;
    }

    return normal;
}

vec3 occludeAmbientColor(vec3 ambientColor, TextureChannels textureChannels) {
    float sampledOcclusion = 0.0;
    if (textureChannels.occlusion != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.occlusion);
        sampledOcclusion = texture(occlusionSampler, uv).r;
    }
    return mix(ambientColor, ambientColor * sampledOcclusion, material.occlusion);
}

uint getAlphaMode() {
    return (material.occlusionTextureChannelAlphaModeAndUnlitFlag >> 16) & 255;
}

bool isMasked(vec4 baseColor) {
    return getAlphaMode() == ALPHA_MODE_MASK && baseColor.a < material.alphaCutoff;
}

float getAlpha(vec4 baseColor) {
    if (getAlphaMode() == ALPHA_MODE_BLEND) {
        return baseColor.a;
    }
    return 1.0;
}

bool isUnlit() {
    uint unlitFlag = (material.occlusionTextureChannelAlphaModeAndUnlitFlag >> 8) & 255;
    if (unlitFlag == UNLIT_FLAG_UNLIT) {
        return true;
    }
    return false;
}

vec3 f(vec3 f0, vec3 v, vec3 h) {
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
}

vec3 f(vec3 f0, vec3 v, vec3 n, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - max(dot(v, n), 0.0), 5.0);
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

float computeAttenuation(float distance, float range) {
    if (range < 0.0) {
        return 1.0;
    }
    return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / pow(distance, 2.0);
}

vec3 computeColor(vec3 baseColor, float metallic, float roughness, vec3 n, vec3 l, vec3 v, vec3 h, vec3 lightColor, float lightIntensity) {
    vec3 color = vec3(0.0);
    if (dot(n, l) > 0.0 || dot(n, v) > 0.0) {
        vec3 cDiffuse = mix(baseColor * (1.0 - DIELECTRIC_SPECULAR.r), BLACK, metallic);
        vec3 f0 = mix(DIELECTRIC_SPECULAR, baseColor, metallic);
        float a = roughness * roughness;

        vec3 f = f(f0, v, h);
        float vis = vis(n, l, v, a);
        float d = d(a, n, h);

        vec3 diffuse = cDiffuse / PI;
        vec3 fDiffuse = (1 - f) * diffuse;
        vec3 fSpecular = max(f * vis * d, 0.0);
        color = max(dot(n, l), 0.0) * (fDiffuse + fSpecular) * lightColor * lightIntensity;
    }
    return color;
}

vec3 computeDirectionalLight(Light light, vec3 baseColor, float metallic, float roughness, vec3 n, vec3 v) {
    vec3 l = -normalize(light.direction.xyz);
    vec3 h = normalize(l + v);
    return computeColor(baseColor.rgb, metallic, roughness, n, l, v, h, light.color.rgb, light.intensity);
}

vec3 computePointLight(Light light, vec3 baseColor, float metallic, float roughness, vec3 n, vec3 v) {
    vec3 toLight = light.position.xyz - oPositions;
    float distance = length(toLight);
    vec3 l = normalize(toLight);
    vec3 h = normalize(l + v);

    float attenuation = computeAttenuation(distance, light.range);

    return computeColor(baseColor.rgb, metallic, roughness, n, l, v, h, light.color.rgb, light.intensity * attenuation);
}

vec3 computeSpotLight(Light light, vec3 baseColor, float metallic, float roughness, vec3 n, vec3 v) {
    vec3 invLightDir = -normalize(light.direction.xyz);

    vec3 toLight = light.position.xyz - oPositions;
    float distance = length(toLight);
    vec3 l = normalize(toLight);
    vec3 h = normalize(l + v);

    float attenuation = computeAttenuation(distance, light.range);

    float cd = dot(invLightDir, l);
    float angularAttenuation = max(0.0, cd * light.angleScale + light.angleOffset);
    angularAttenuation *= angularAttenuation;

    return computeColor(baseColor.rgb, metallic, roughness, n, l, v, h, light.color.rgb, light.intensity * attenuation * angularAttenuation);
}

vec3 prefilteredReflectionLinear(vec3 R, float roughness) {
	float lod = roughness * MAX_REFLECTION_LOD;
	float lodf = floor(lod);
	float lodc = ceil(lod);
	vec3 a = textureLod(preFilteredSampler, R, lodf).rgb;
	vec3 b = textureLod(preFilteredSampler, R, lodc).rgb;
	return mix(a, b, lod - lodf);
}

vec3 prefilteredReflection(vec3 R, float roughness) {
	float lod = roughness * MAX_REFLECTION_LOD;
	return textureLod(preFilteredSampler, R, lod).rgb;
}

vec3 computeIBL(vec3 baseColor, vec3 v, vec3 n, float metallic, float roughness) {
    vec3 f0 = mix(DIELECTRIC_SPECULAR, baseColor, metallic);
    vec3 f = f(f0, v, n, roughness);
    vec3 kD = 1.0 - f;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMapSampler, n).rgb;
    vec3 diffuse = irradiance * baseColor.rgb;

    vec3 r = normalize(reflect(-v, n));
    vec3 reflection = prefilteredReflection(r, roughness);
    vec2 envBRDF = texture(brdfLookupSampler, vec2(max(dot(n, v), 0.0), roughness)).rg;
    vec3 specular = reflection * (f * envBRDF.x + envBRDF.y);

    return kD * diffuse + specular;
}

void main() {
    TextureChannels textureChannels = getTextureChannels();

    vec4 baseColor = getBaseColor(textureChannels);
    if (isMasked(baseColor)) {
        discard;
    }
    float alpha = getAlpha(baseColor);

    if (isUnlit()) {
        outColor = vec4(baseColor.rgb, alpha);
        return;
    }

    float metallic = getMetallic(textureChannels);
    float roughness = getRoughness(textureChannels);
    vec3 emissive = getEmissiveColor(textureChannels);

    vec3 n = getNormal(textureChannels);
    vec3 v = normalize(cameraUBO.eye - oPositions);

    vec3 color = vec3(0.0);

    for (int i = 0; i < LIGHT_COUNT; i++) {

        Light light = lights.lights[i];
        uint lightType = light.type;

        if (lightType == DIRECTIONAL_LIGHT_TYPE) {
            color += computeDirectionalLight(light, baseColor.rgb, metallic, roughness, n, v);
        } else if (lightType == POINT_LIGHT_TYPE) {
            color += computePointLight(light, baseColor.rgb, metallic, roughness, n, v);
        } else if (lightType == SPOT_LIGHT_TYPE) {
            color += computeSpotLight(light, baseColor.rgb, metallic, roughness, n, v);
        }
    }

    vec3 ambient = computeIBL(baseColor.rgb, v, n, metallic, roughness);

    color += emissive + occludeAmbientColor(ambient, textureChannels);

    outColor = vec4(color, alpha);

#ifdef DEBUG_COLOR
    outColor = vec4(baseColor.rgb, 1.0);
#endif

#ifdef DEBUG_EMISSIVE
    outColor = vec4(pow(emissive, vec3(1.0/2.2)), 1.0);
#endif

#ifdef DEBUG_METALLIC
    outColor = vec4(vec3(metallic), 1.0);
#endif

#ifdef DEBUG_ROUGHNESS
    outColor = vec4(vec3(roughness), 1.0);
#endif

#ifdef DEBUG_OCCLUSION
    outColor = vec4(occludeAmbientColor(vec3(1.0), textureChannels), 1.0);
#endif

#ifdef DEBUG_NORMAL
    outColor = vec4(n*0.5 + 0.5, 1.0);
#endif

#ifdef DEBUG_ALPHA
     outColor = vec4(vec3(baseColor.a), 1.0);
#endif

#ifdef DEBUG_UVS
     outColor = vec4(vec2(oTexcoords), 0.0, 1.0);
#endif
}
