#version 450
#extension GL_GOOGLE_include_directive : require

#include "libs/camera.glsl"
#include "libs/material.glsl"

// -- Constants --
layout(constant_id = 0) const uint MAX_LIGHT_COUNT = 1;
layout(constant_id = 1) const uint MAX_REFLECTION_LOD = 1;
layout(constant_id = 2) const uint PASS = 0;

const uint PASS_OPAQUE = 0;
const uint PASS_OPAQUE_TRANSPARENT = 1;
const uint PASS_TRANSPARENT = 2;

const uint OUTPUT_MODE_FINAL = 0;
const uint OUTPUT_MODE_COLOR = 1;
const uint OUTPUT_MODE_EMISSIVE = 2;
const uint OUTPUT_MODE_METALLIC = 3;
const uint OUTPUT_MODE_SPECULAR = 4;
const uint OUTPUT_MODE_ROUGHNESS = 5;
const uint OUTPUT_MODE_OCCLUSION = 6;
const uint OUTPUT_MODE_NORMAL = 7;
const uint OUTPUT_MODE_ALPHA = 8;
const uint OUTPUT_MODE_UVS0 = 9;
const uint OUTPUT_MODE_UVS1 = 10;
const uint OUTPUT_MODE_SSAO = 11;
const uint OUTPUT_MODE_CLEARCOAT_FACTOR = 12;
const uint OUTPUT_MODE_CLEARCOAT_ROUGHNESS = 13;
const uint OUTPUT_MODE_CLEARCOAT_NORMAL = 14;

const vec3 CLEARCOAT_DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;

const uint DIRECTIONAL_LIGHT_TYPE = 0;
const uint POINT_LIGHT_TYPE = 1;
const uint SPOT_LIGHT_TYPE = 2;

// -- Structures --
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

struct PbrInfo {
    vec3 baseColor;
    float metallic;
    vec3 specular;
    float roughness;
    bool metallicRoughnessWorkflow;
    vec3 normal;
    float clearcoatFactor;
    float clearcoatRoughness;
    vec3 clearcoatNormal;
    vec3 dielectricSpecular;
};

struct Config {
    uint outputMode;
    float emissiveIntensity;
};

// -- Inputs --
layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords0;
layout(location = 2) in vec2 oTexcoords1;
layout(location = 3) in vec3 oPositions;
layout(location = 4) in vec4 oColors;
layout(location = 5) in mat3 oTBN;

// -- Descriptors --
layout(binding = 0, set = 0) uniform CameraUBO {
    Camera camera;
};

layout(binding = 1, set = 0) uniform ConfigUBO {
    Config config;
};

layout(binding = 2, set = 0) uniform LightsUBO {
    uint count;
    Light lights[MAX_LIGHT_COUNT];
} lights;

layout(binding = 5, set = 1) uniform samplerCube irradianceMapSampler;
layout(binding = 6, set = 1) uniform samplerCube preFilteredSampler;
layout(binding = 7, set = 1) uniform sampler2D brdfLookupSampler;
layout(binding = 8, set = 2) uniform sampler2D colorSampler;
layout(binding = 9, set = 2) uniform sampler2D normalsSampler;
// This sampler contains either:
// - metallic (b) + glossiness (g) for metallic/roughness workflow
// - specular (rgb) + glossiness (a) for specular/glossiness workflow
layout(binding = 10, set = 2) uniform sampler2D materialSampler;
layout(binding = 11, set = 2) uniform sampler2D occlusionSampler;
layout(binding = 12, set = 2) uniform sampler2D emissiveSampler;
layout(binding = 13, set = 2) uniform sampler2D clearcoatFactorSampler;
layout(binding = 14, set = 2) uniform sampler2D clearcoatRoughnessSampler;
layout(binding = 15, set = 2) uniform sampler2D clearcoatNormalSampler;
layout(binding = 16, set = 3) uniform sampler2D aoMapSampler;

layout(binding = 17, set = 4) uniform MaterialUBO {
    Material material;
};

// Output
layout(location = 0) out vec4 outColor;

vec2 getUV(uint texChannel, mat4 transform) {
    vec2 uv = texChannel == 0 ? oTexcoords0 : oTexcoords1;
    return (transform * vec4(uv, 1.0, 1.0)).xy;
}

vec4 getBaseColor() {
    vec4 color = material.color;
    if(material.colorTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.colorTextureChannel, material.colorTextureTransform);
        color *= texture(colorSampler, uv);
    }
    return color * oColors;
}

float getMetallic() {
    float metallic = material.metallicSpecular.r;
    if(material.materialTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.materialTextureChannel, material.materialTextureTransform);
        metallic *= texture(materialSampler, uv).b;
    }
    return metallic;
}

vec3 getSpecular() {
    vec3 specular = material.metallicSpecular;
    if(material.materialTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.materialTextureChannel, material.materialTextureTransform);
        specular *= texture(materialSampler, uv).rgb;
    }
    return specular;
}

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular) {
    const float c_MinRoughness = 0.04;
    float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
    float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
    if (perceivedSpecular < c_MinRoughness) {
        return 0.0;
    }
    float a = c_MinRoughness;
    float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
    float c = c_MinRoughness - perceivedSpecular;
    float D = max(b * b - 4.0 * a * c, 0.0);
    return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

float getRoughness(bool metallicRoughnessWorkflow) {
    float roughness = material.roughnessGlossiness;
    if(material.materialTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.materialTextureChannel, material.materialTextureTransform);
        if (metallicRoughnessWorkflow) {
            roughness *= texture(materialSampler, uv).g;
        } else {
            roughness *= texture(materialSampler, uv).a;
        }
    }

    if (metallicRoughnessWorkflow) {
        return roughness;
    }
    return (1 - roughness);
}

vec3 getEmissiveColor() {
    vec3 emissive = material.emissiveFactor;
    if(material.emissiveTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.emissiveTextureChannel, material.emissiveTextureTransform);
        emissive *= texture(emissiveSampler, uv).rgb;
    }
    return emissive * config.emissiveIntensity;
}

vec3 getNormal() {
    vec3 normal = normalize(oNormals);
    if (material.normalsTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.normalsTextureChannel, material.normalsTextureTransform);
        vec3 normalMap = texture(normalsSampler, uv).rgb * 2.0 - 1.0;
        normal = normalize(oTBN * normalMap);
    }
    
    if (!gl_FrontFacing) {
        normal *= -1.0;
    }

    return normal;
}

float sampleAOMap() {
    ivec2 size = textureSize(aoMapSampler, 0);
    vec2 coords = vec2(float(gl_FragCoord.x) / float(size.x), float(gl_FragCoord.y) / float(size.y));
    return texture(aoMapSampler, coords).r;
}

vec3 occludeAmbientColor(vec3 ambientColor) {
    float aoMapSample = sampleAOMap();
    float sampledOcclusion = 0.0;
    if (material.occlusionTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.occlusionTextureChannel, material.occlusionTextureTransform);
        sampledOcclusion = texture(occlusionSampler, uv).r;
    }
    return mix(ambientColor, ambientColor * sampledOcclusion, material.occlusion) * aoMapSample;
}

uint getAlphaMode() {
    return material.alphaMode;
}

bool isMasked(vec4 baseColor) {
    // discard masked fragments
    if (PASS == PASS_OPAQUE) {
        float alphaCutoff = material.alphaCutoff;
        return getAlphaMode() == ALPHA_MODE_MASK && baseColor.a + ALPHA_CUTOFF_BIAS < alphaCutoff;
    }

    // discard non opaque fragment
    if (PASS == PASS_OPAQUE_TRANSPARENT) {
        return baseColor.a < (1.0 - ALPHA_CUTOFF_BIAS);
    }

    // discard opaque fragments
    if (PASS == PASS_TRANSPARENT) {
        return baseColor.a > (1.0 - ALPHA_CUTOFF_BIAS);
    }

    // no supposed to happen, so discard so it is noticable if it ever does
    return true;
}

float getAlpha(vec4 baseColor) {
    if (getAlphaMode() == ALPHA_MODE_BLEND) {
        return baseColor.a;
    }
    return 1.0;
}

bool isUnlit() {
    return material.isUnlit;
}

bool isMetallicRoughnessWorkflow() {
    uint workflow = material.workflow;
    if (workflow == METALLIC_ROUGHNESS_WORKFLOW) {
        return true;
    }
    return false;
}

float getClearcoatFactor() {
    float factor = material.clearcoatFactor;
    if(material.clearcoatFactorTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.clearcoatFactorTextureChannel, material.clearcoatFactorTextureTransform);
        factor *= texture(clearcoatFactorSampler, uv).r;
    }
    return factor;
}

float getClearcoatRoughness() {
    float roughness = material.clearcoatRoughness;
    if(material.clearcoatRoughnessTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.clearcoatRoughnessTextureChannel, material.clearcoatRoughnessTextureTransform);
        roughness *= texture(clearcoatRoughnessSampler, uv).g;
    }
    return roughness;
}

vec3 getClearcoatNormal() {
    vec3 normal = normalize(oNormals);
    if (material.clearcoatNormalsTextureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(material.clearcoatNormalsTextureChannel, material.clearcoatNormalsTextureTransform);
        vec3 normalMap = texture(clearcoatNormalSampler, uv).rgb * 2.0 - 1.0;
        normal = normalize(oTBN * normalMap);
    }
    
    if (!gl_FrontFacing) {
        normal *= -1.0;
    }

    return normal;
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

vec3 computeColor(
    PbrInfo pbrInfo,
    vec3 l,
    vec3 v,
    vec3 h,
    vec3 lightColor,
    float lightIntensity
) {
    vec3 n = pbrInfo.normal;
    
    vec3 color = vec3(0.0);
    if (dot(n, l) > 0.0 || dot(n, v) > 0.0) {
        vec3 cDiffuse;
        vec3 f0;
        if (pbrInfo.metallicRoughnessWorkflow) {
            cDiffuse = mix(pbrInfo.baseColor * (1.0 - pbrInfo.dielectricSpecular.r), BLACK, pbrInfo.metallic);
            f0 = mix(pbrInfo.dielectricSpecular, pbrInfo.baseColor, pbrInfo.metallic);
        } else {
            cDiffuse = pbrInfo.baseColor * (1.0 - max(pbrInfo.specular.r, max(pbrInfo.specular.g, pbrInfo.specular.b)));
            f0 = pbrInfo.specular;
        }

        float a = pbrInfo.roughness * pbrInfo.roughness;

        vec3 f = f(f0, v, h);
        float vis = vis(n, l, v, a);
        float d = d(a, n, h);

        vec3 diffuse = cDiffuse / PI;
        vec3 fDiffuse = (1 - f) * diffuse;
        vec3 fSpecular = max(f * vis * d, 0.0);
        color = max(dot(n, l), 0.0) * (fDiffuse + fSpecular) * lightColor * lightIntensity;
    }

    n = pbrInfo.clearcoatNormal;
    if (dot(n, l) > 0.0 || dot(n, v) > 0.0) {
        float a = pbrInfo.clearcoatRoughness * pbrInfo.clearcoatRoughness;

        vec3 f = f(CLEARCOAT_DIELECTRIC_SPECULAR, v, h);
        float vis = vis(n, l, v, a);
        float d = d(a, n, h);

        vec3 fSpecular = max(f * vis * d, 0.0) * pbrInfo.clearcoatFactor;

        color = color * (1.0 - fSpecular) + (fSpecular * lightColor * lightIntensity);
    }

    return color;
}

vec3 computeDirectionalLight(Light light, PbrInfo pbrInfo, vec3 v) {
    vec3 l = -normalize(light.direction.xyz);
    vec3 h = normalize(l + v);
    return computeColor(pbrInfo, l, v, h, light.color.rgb, light.intensity);
}

vec3 computePointLight(Light light, PbrInfo pbrInfo, vec3 v) {
    vec3 toLight = light.position.xyz - oPositions;
    float distance = length(toLight);
    vec3 l = normalize(toLight);
    vec3 h = normalize(l + v);

    float attenuation = computeAttenuation(distance, light.range);

    return computeColor(pbrInfo, l, v, h, light.color.rgb, light.intensity * attenuation);
}

vec3 computeSpotLight(Light light, PbrInfo pbrInfo, vec3 v) {
    vec3 invLightDir = -normalize(light.direction.xyz);

    vec3 toLight = light.position.xyz - oPositions;
    float distance = length(toLight);
    vec3 l = normalize(toLight);
    vec3 h = normalize(l + v);

    float attenuation = computeAttenuation(distance, light.range);

    float cd = dot(invLightDir, l);
    float angularAttenuation = max(0.0, cd * light.angleScale + light.angleOffset);
    angularAttenuation *= angularAttenuation;

    return computeColor(pbrInfo, l, v, h, light.color.rgb, light.intensity * attenuation * angularAttenuation);
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

vec3 computeIBL(PbrInfo pbrInfo, vec3 v) {

    vec3 f0 = pbrInfo.specular;
    if (pbrInfo.metallicRoughnessWorkflow) {
        f0 = mix(pbrInfo.dielectricSpecular, pbrInfo.baseColor, pbrInfo.metallic);
    }

    vec3 n = pbrInfo.normal;
    vec3 fBase = f(f0, v, n, pbrInfo.roughness);
    vec3 kD = 1.0 - fBase;
    kD *= 1.0 - pbrInfo.metallic;

    vec3 irradiance = texture(irradianceMapSampler, n).rgb;
    vec3 diffuse = irradiance * pbrInfo.baseColor;

    vec3 r = normalize(reflect(-v, n));
    vec3 reflection = prefilteredReflection(r, pbrInfo.roughness);
    vec2 envBRDF = texture(brdfLookupSampler, vec2(max(dot(n, v), 0.0), pbrInfo.roughness)).rg;
    vec3 specular = reflection * (fBase * envBRDF.x + envBRDF.y);

    // clearcoat layer
    vec3 cn = pbrInfo.clearcoatNormal;
    vec3 fClearcoat = f(CLEARCOAT_DIELECTRIC_SPECULAR, v, cn, pbrInfo.clearcoatRoughness);

    vec3 cr = normalize(reflect(-v, cn));
    vec3 cReflection = prefilteredReflection(cr, pbrInfo.clearcoatRoughness);
    vec2 cEnvBRDF = texture(brdfLookupSampler, vec2(max(dot(cn, v), 0.0), pbrInfo.clearcoatRoughness)).rg;
    vec3 cSpecular = cReflection * (fClearcoat * cEnvBRDF.x + cEnvBRDF.y);

    // final color
    vec3 baseLayer = kD * diffuse + specular;
    vec3 clearcoat = cSpecular * pbrInfo.clearcoatFactor;
    
    return baseLayer * (1.0 - fClearcoat * pbrInfo.clearcoatFactor) + clearcoat; 
}

void main() {
    vec4 baseColor = getBaseColor();
    if (isMasked(baseColor)) {
        discard;
    }
    float alpha = getAlpha(baseColor);

    if (isUnlit()) {
        outColor = vec4(baseColor.rgb, alpha);
        return;
    }

    bool metallicRoughnessWorkflow = isMetallicRoughnessWorkflow();
    vec3 specular = getSpecular();
    float roughness = getRoughness(metallicRoughnessWorkflow);

    float metallic;
    if (metallicRoughnessWorkflow) {
        metallic = getMetallic();
    } else {
        float maxSpecular = max(specular.r, max(specular.g, specular.b));
        metallic = convertMetallic(baseColor.rgb, specular, maxSpecular);
    }

    vec3 emissive = getEmissiveColor();

    float clearcoatFactor = getClearcoatFactor();
    float clearcoatRoughness = getClearcoatRoughness();
    vec3 clearcoatNormal = getClearcoatNormal();

    float dielectricSpecularSqrt = (material.ior - 1)/(material.ior + 1);
    float dielectricSpecular = dielectricSpecularSqrt*dielectricSpecularSqrt;

    vec3 n = getNormal();
    vec3 v = normalize(camera.eye.xyz - oPositions);

    PbrInfo pbrInfo = PbrInfo(
        baseColor.rgb, 
        metallic, 
        specular, 
        roughness, 
        metallicRoughnessWorkflow, 
        n,
        clearcoatFactor,
        clearcoatRoughness,
        clearcoatNormal,
        vec3(dielectricSpecular)
    );

    vec3 color = vec3(0.0);

    for (int i = 0; i < lights.count; i++) {

        Light light = lights.lights[i];
        uint lightType = light.type;

        if (lightType == DIRECTIONAL_LIGHT_TYPE) {
            color += computeDirectionalLight(light, pbrInfo, v);
        } else if (lightType == POINT_LIGHT_TYPE) {
            color += computePointLight(light, pbrInfo, v);
        } else if (lightType == SPOT_LIGHT_TYPE) {
            color += computeSpotLight(light, pbrInfo, v);
        }
    }

    vec3 ambient = computeIBL(pbrInfo, v);

    color += emissive + occludeAmbientColor(ambient);

    uint outputMode = config.outputMode;
    if (outputMode == OUTPUT_MODE_FINAL) {
        outColor = vec4(color, alpha);
    } else if (outputMode == OUTPUT_MODE_COLOR) {
        outColor = vec4(baseColor.rgb, 1.0);
    } else if (outputMode == OUTPUT_MODE_EMISSIVE) {
        outColor = vec4(emissive, 1.0);
    } else if (outputMode == OUTPUT_MODE_METALLIC) {
        outColor = vec4(vec3(metallic), 1.0);
    } else if (outputMode == OUTPUT_MODE_SPECULAR) {
        outColor = vec4(specular, 1.0);
    } else if (outputMode == OUTPUT_MODE_ROUGHNESS) {
        outColor = vec4(vec3(roughness), 1.0);
    } else if (outputMode == OUTPUT_MODE_OCCLUSION) {
        outColor = vec4(occludeAmbientColor(vec3(1.0)), 1.0);
    } else if (outputMode == OUTPUT_MODE_NORMAL) {
        outColor = vec4(n*0.5 + 0.5, 1.0);
    } else if (outputMode == OUTPUT_MODE_ALPHA) {
        outColor = vec4(vec3(baseColor.a), 1.0);
    } else if (outputMode == OUTPUT_MODE_UVS0) {
        outColor = vec4(vec2(oTexcoords0), 0.0, 1.0);
    } else if (outputMode == OUTPUT_MODE_UVS1) {
        outColor = vec4(vec2(oTexcoords1), 0.0, 1.0);
    } else if (outputMode == OUTPUT_MODE_SSAO) {
        float ao = sampleAOMap();
        outColor = vec4(vec3(ao), 1.0);
    } else if (outputMode == OUTPUT_MODE_CLEARCOAT_FACTOR) {
        outColor = vec4(vec3(clearcoatFactor), 1.0);
    } else if (outputMode == OUTPUT_MODE_CLEARCOAT_ROUGHNESS) {
        outColor = vec4(vec3(clearcoatRoughness), 1.0);
    } else if (outputMode == OUTPUT_MODE_CLEARCOAT_NORMAL) {
        outColor = outColor = vec4(clearcoatNormal*0.5 + 0.5, 1.0);
    }
}
