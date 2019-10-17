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

layout(constant_id = 0) const uint MAX_DIRECTIONAL_LIGHTS = 1;
layout(constant_id = 1) const uint MAX_POINT_LIGHTS = 1;
const vec3 DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;
const uint NO_TEXTURE_ID = 255;
const uint ALPHA_MODE_MASK = 1;
const uint ALPHA_MODE_BLEND = 2;
const float MAX_REFLECTION_LOD = 9.0; // last mip mips for 512 px res TODO: specializations ?

struct TextureIds {
    uint color;
    uint metallicRoughness;
    uint emissive;
    uint normal;
    uint occlusion;
};

struct DirectionalLight {
    vec4 direction;
    vec4 color;
    float intensity;
};

struct PointLight {
    vec4 position;
    vec4 color;
    float intensity;
    float range;
};

layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords;
layout(location = 2) in vec3 oPositions;
layout(location = 3) in vec4 oColors;
layout(location = 4) in mat3 oTBN;

layout(push_constant) uniform Material {
    vec4 color;
    vec4 emissiveAndRoughness;
    float metallic;
    float occlusion;
    // Contains the texture ids for color, metallic/roughness, emissive and normal (each taking 8 bytes)
    uint colorMetallicRoughnessEmissiveNormalTextureIds;
    // Contains the occlusion texture id and the alpha mode (each taking 8 bytes) + 16 bytes of right padding
    uint occlusionTextureIdAndAlphaMode;
    float alphaCutoff;
} material;

// -- Descriptors --
layout(binding = 0, set = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    vec3 eye;    
} cameraUBO;
layout(binding = 1, set = 0) uniform DirectionalLights {
    DirectionalLight lights[MAX_DIRECTIONAL_LIGHTS];
} directionalLights;
layout(binding = 2, set = 0) uniform PointLights {
    PointLight lights[MAX_POINT_LIGHTS];
} pointLights;
layout(binding = 5, set = 1) uniform samplerCube irradianceMapSampler;
layout(binding = 6, set = 1) uniform samplerCube preFilteredSampler;
layout(binding = 7, set = 1) uniform sampler2D brdfLookupSampler;
layout(binding = 8, set = 2) uniform sampler2D colorSampler;
layout(binding = 9, set = 2) uniform sampler2D normalsSampler;
layout(binding = 10, set = 2) uniform sampler2D metallicRoughnessSampler;
layout(binding = 11, set = 2) uniform sampler2D occlusionSampler;
layout(binding = 12, set = 2) uniform sampler2D emissiveSampler;

// Output
layout(location = 0) out vec4 outColor;

TextureIds getTextureIds() {
    return TextureIds(
        (material.colorMetallicRoughnessEmissiveNormalTextureIds >> 24) & 255,
        (material.colorMetallicRoughnessEmissiveNormalTextureIds >> 16) & 255,
        (material.colorMetallicRoughnessEmissiveNormalTextureIds >> 8) & 255,
        material.colorMetallicRoughnessEmissiveNormalTextureIds & 255,
        (material.occlusionTextureIdAndAlphaMode >> 24) & 255
    );
}

vec4 getBaseColor(TextureIds textureIds) {
    vec4 color = material.color;
    if(textureIds.color != NO_TEXTURE_ID) {
        vec4 sampledColor= texture(colorSampler, oTexcoords);
        color *= vec4(pow(sampledColor.rgb, vec3(2.2)), sampledColor.a);
    }
    return color * oColors;
}

float getMetallic(TextureIds textureIds) {
    float metallic = material.metallic;
    if(textureIds.metallicRoughness != NO_TEXTURE_ID) {
        metallic *= texture(metallicRoughnessSampler, oTexcoords).b;
    }
    return metallic;
}

float getRoughness(TextureIds textureIds) {
    float roughness = material.emissiveAndRoughness.a;
    if(textureIds.metallicRoughness != NO_TEXTURE_ID) {
        roughness *= texture(metallicRoughnessSampler, oTexcoords).g;
    }
    return roughness;
}

vec3 getEmissiveColor(TextureIds textureIds) {
    vec3 emissive = material.emissiveAndRoughness.rgb;
    if(textureIds.emissive != NO_TEXTURE_ID) {
        emissive *= pow(texture(emissiveSampler, oTexcoords).rgb, vec3(2.2));
    }
    return emissive;
}

vec3 getNormal(TextureIds textureIds) {
    if (textureIds.normal != NO_TEXTURE_ID) {
        vec3 normal = texture(normalsSampler, oTexcoords).rgb * 2.0 - 1.0;
        return normalize(oTBN * normal);
    }
    return normalize(oNormals);
}

vec3 occludeAmbientColor(vec3 ambientColor, TextureIds textureIds) {
    float sampledOcclusion = 0.0;
    if (textureIds.occlusion != NO_TEXTURE_ID) {
        sampledOcclusion = texture(occlusionSampler, oTexcoords).r;
    }
    return mix(ambientColor, ambientColor * sampledOcclusion, material.occlusion);
}

uint getAlphaMode() {
    return (material.occlusionTextureIdAndAlphaMode >> 16) & 255;
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
    TextureIds textureIds = getTextureIds();

    vec4 baseColor = getBaseColor(textureIds);
    if (isMasked(baseColor)) {
        discard;
    }
    float alpha = getAlpha(baseColor);

    float metallic = getMetallic(textureIds);
    float roughness = getRoughness(textureIds);
    vec3 emissive = getEmissiveColor(textureIds);

    vec3 n = getNormal(textureIds);
    vec3 v = normalize(cameraUBO.eye - oPositions);

    vec3 color = vec3(0.0);

    // Directional lights
    for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
        float lightIntensity = directionalLights.lights[i].intensity;
        vec3 lightColor = directionalLights.lights[i].color.rgb;
        vec3 l = -normalize(directionalLights.lights[i].direction.xyz);
        vec3 h = normalize(l + v);
        color += computeColor(baseColor.rgb, metallic, roughness, n, l, v, h, lightColor, lightIntensity);
    }

    // Point lights
    for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
        float lightIntensity = pointLights.lights[i].intensity;
        vec3 lightColor = pointLights.lights[i].color.rgb;
        vec3 lightPosition = pointLights.lights[i].position.xyz;
        float lightRange = pointLights.lights[i].range;

        vec3 toLight = lightPosition - oPositions;
        float distance = length(toLight);
        vec3 l = normalize(toLight);
        vec3 h = normalize(l + v);

        float attenuation = 1.0;
        if (lightRange >= 0.0) {
            attenuation = max(min(1.0 - pow(distance / lightRange, 4.0), 1.0), 0.0) / pow(distance, 2.0);
        }

        color += computeColor(baseColor.rgb, metallic, roughness, n, l, v, h, lightColor, lightIntensity * attenuation);
    }

    vec3 ambient = computeIBL(baseColor.rgb, v, n, metallic, roughness);

    color += emissive + occludeAmbientColor(ambient, textureIds);

    color = color/(color + 1.0);
    color = pow(color, vec3(1.0/2.2));
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
    outColor = vec4(occludeAmbientColor(vec3(1.0), textureIds), 1.0);
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
