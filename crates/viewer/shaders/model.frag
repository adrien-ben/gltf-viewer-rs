#version 450

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

const vec3 DIELECTRIC_SPECULAR = vec3(0.04);
const vec3 BLACK = vec3(0.0);
const float PI = 3.14159;

const uint NO_TEXTURE_ID = 3;

const uint ALPHA_MODE_MASK = 1;
const uint ALPHA_MODE_BLEND = 2;
const float ALPHA_CUTOFF_BIAS = 0.0000001;

const uint DIRECTIONAL_LIGHT_TYPE = 0;
const uint POINT_LIGHT_TYPE = 1;
const uint SPOT_LIGHT_TYPE = 2;

const uint UNLIT_FLAG_UNLIT = 1;

const uint METALLIC_ROUGHNESS_WORKFLOW = 0;

// -- Structures --
struct TextureChannels {
    uint color;
    uint material;
    uint emissive;
    uint normal;
    uint occlusion;
    uint clearcoatFactor;
    uint clearcoatRoughness;
    uint clearcoatNormal;
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
};

// -- Inputs --
layout(location = 0) in vec3 oNormals;
layout(location = 1) in vec2 oTexcoords0;
layout(location = 2) in vec2 oTexcoords1;
layout(location = 3) in vec3 oPositions;
layout(location = 4) in vec4 oColors;
layout(location = 5) in mat3 oTBN;

// -- Push constants
layout(push_constant) uniform MaterialUniform {
    vec4 color;
    vec3 emissiveFactor;
    // - roughness for metallic/roughness workflows
    // - glossiness for specular/glossiness workflows
    float roughnessGlossiness;
    // Contains the metallic (or specular) factor.
    // - metallic: r (for metallic/roughness workflows)
    // - specular: rgb (for specular/glossiness workflows)
    vec3 metallicSpecular;
    float occlusion;
    float alphaCutoff;
    float clearcoatFactor;
    float clearcoatRoughness;
    // Contains the texture channels. Each channel taking 2 bits
    // [0-1] Color texture channel
    // [2-3] metallic/roughness or specular/glossiness texture channel
    // [4-5] emissive texture channel
    // [6-7] normals texture channel
    // [8-9] Occlusion texture channel
    // [10-11] Clearcoat factor texture channel
    // [12-13] Clearcoat roughness texture channel
    // [14-15] Clearcoat normal texture channel
    // [16-31] Reserved
    uint texturesChannels;
    // Contains alpha mode, unlit flag and workflow flag
    // [0-7] Alpha mode
    // [8-15] Unlit flag
    // [16-23] Workflow (metallic/roughness or specular/glossiness)
    // [24-31] Reserved
    uint alphaModeUnlitFlagAndWorkflow;
    uint lightCount;
    uint outputMode;
    float emissiveIntensity;
} material;

// -- Descriptors --
layout(binding = 0, set = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 invertedProj;
    vec4 eye;
    float zNear;
    float zFar;
} cameraUBO;
layout(binding = 1, set = 0) uniform Lights {
    Light lights[MAX_LIGHT_COUNT];
} lights;
layout(binding = 4, set = 1) uniform samplerCube irradianceMapSampler;
layout(binding = 5, set = 1) uniform samplerCube preFilteredSampler;
layout(binding = 6, set = 1) uniform sampler2D brdfLookupSampler;
layout(binding = 7, set = 2) uniform sampler2D colorSampler;
layout(binding = 8, set = 2) uniform sampler2D normalsSampler;
// This sampler contains either:
// - metallic (b) + glossiness (g) for metallic/roughness workflow
// - specular (rgb) + glossiness (a) for specular/glossiness workflow
layout(binding = 9, set = 2) uniform sampler2D materialSampler;
layout(binding = 10, set = 2) uniform sampler2D occlusionSampler;
layout(binding = 11, set = 2) uniform sampler2D emissiveSampler;
layout(binding = 12, set = 2) uniform sampler2D clearcoatFactorSampler;
layout(binding = 13, set = 2) uniform sampler2D clearcoatRoughnessSampler;
layout(binding = 14, set = 2) uniform sampler2D clearcoatNormalSampler;
layout(binding = 15, set = 3) uniform sampler2D aoMapSampler;

// Output
layout(location = 0) out vec4 outColor;

TextureChannels getTextureChannels() {
    return TextureChannels(
        (material.texturesChannels >> 30) & 3,
        (material.texturesChannels >> 28) & 3,
        (material.texturesChannels >> 26) & 3,
        (material.texturesChannels >> 24) & 3,
        (material.texturesChannels >> 22) & 3,
        (material.texturesChannels >> 20) & 3,
        (material.texturesChannels >> 18) & 3,
        (material.texturesChannels >> 16) & 3
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
        color *= texture(colorSampler, uv);
    }
    return color * oColors;
}

float getMetallic(TextureChannels textureChannels) {
    float metallic = material.metallicSpecular.r;
    if(textureChannels.material != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.material);
        metallic *= texture(materialSampler, uv).b;
    }
    return metallic;
}

vec3 getSpecular(TextureChannels textureChannels) {
    vec3 specular = material.metallicSpecular;
    if(textureChannels.material != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.material);
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

float getRoughness(TextureChannels textureChannels, bool metallicRoughnessWorkflow) {
    float roughness = material.roughnessGlossiness;
    if(textureChannels.material != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.material);
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

vec3 getEmissiveColor(TextureChannels textureChannels) {
    vec3 emissive = material.emissiveFactor;
    if(textureChannels.emissive != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.emissive);
        emissive *= texture(emissiveSampler, uv).rgb;
    }
    return emissive * material.emissiveIntensity;
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

float sampleAOMap() {
    ivec2 size = textureSize(aoMapSampler, 0);
    vec2 coords = vec2(float(gl_FragCoord.x) / float(size.x), float(gl_FragCoord.y) / float(size.y));
    return texture(aoMapSampler, coords).r;
}

vec3 occludeAmbientColor(vec3 ambientColor, TextureChannels textureChannels) {
    float aoMapSample = sampleAOMap();
    float sampledOcclusion = 0.0;
    if (textureChannels.occlusion != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.occlusion);
        sampledOcclusion = texture(occlusionSampler, uv).r;
    }
    return mix(ambientColor, ambientColor * sampledOcclusion, material.occlusion) * aoMapSample;
}

uint getAlphaMode() {
    return (material.alphaModeUnlitFlagAndWorkflow >> 24) & 255;
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
    uint unlitFlag = (material.alphaModeUnlitFlagAndWorkflow >> 16) & 255;
    if (unlitFlag == UNLIT_FLAG_UNLIT) {
        return true;
    }
    return false;
}

bool isMetallicRoughnessWorkflow() {
    uint workflow = (material.alphaModeUnlitFlagAndWorkflow >> 8) & 255;
    if (workflow == METALLIC_ROUGHNESS_WORKFLOW) {
        return true;
    }
    return false;
}

float getClearcoatFactor(TextureChannels textureChannels) {
    float factor = material.clearcoatFactor;
    if(textureChannels.clearcoatFactor != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.clearcoatFactor);
        factor *= texture(clearcoatFactorSampler, uv).r;
    }
    return factor;
}

float getClearcoatRoughness(TextureChannels textureChannels) {
    float roughness = material.clearcoatRoughness;
    if(textureChannels.clearcoatRoughness != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.clearcoatRoughness);
        roughness *= texture(clearcoatRoughnessSampler, uv).g;
    }
    return roughness;
}

vec3 getClearcoatNormal(TextureChannels textureChannels) {
    vec3 normal = normalize(oNormals);
    if (textureChannels.clearcoatNormal != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannels.clearcoatNormal);
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
            cDiffuse = mix(pbrInfo.baseColor * (1.0 - DIELECTRIC_SPECULAR.r), BLACK, pbrInfo.metallic);
            f0 = mix(DIELECTRIC_SPECULAR, pbrInfo.baseColor, pbrInfo.metallic);
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

        vec3 f = f(DIELECTRIC_SPECULAR, v, h);
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
        f0 = mix(DIELECTRIC_SPECULAR, pbrInfo.baseColor, pbrInfo.metallic);
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
    vec3 fClearcoat = f(DIELECTRIC_SPECULAR, v, cn, pbrInfo.clearcoatRoughness);

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

    bool metallicRoughnessWorkflow = isMetallicRoughnessWorkflow();
    vec3 specular = getSpecular(textureChannels);
    float roughness = getRoughness(textureChannels, metallicRoughnessWorkflow);

    float metallic;
    if (metallicRoughnessWorkflow) {
        metallic = getMetallic(textureChannels);
    } else {
        float maxSpecular = max(specular.r, max(specular.g, specular.b));
        metallic = convertMetallic(baseColor.rgb, specular, maxSpecular);
    }

    vec3 emissive = getEmissiveColor(textureChannels);

    float clearcoatFactor = getClearcoatFactor(textureChannels);
    float clearcoatRoughness = getClearcoatRoughness(textureChannels);
    vec3 clearcoatNormal = getClearcoatNormal(textureChannels);

    vec3 n = getNormal(textureChannels);
    vec3 v = normalize(cameraUBO.eye.xyz - oPositions);

    PbrInfo pbrInfo = PbrInfo(
        baseColor.rgb, 
        metallic, 
        specular, 
        roughness, 
        metallicRoughnessWorkflow, 
        n,
        clearcoatFactor,
        clearcoatRoughness,
        clearcoatNormal
    );

    vec3 color = vec3(0.0);

    for (int i = 0; i < material.lightCount; i++) {

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

    color += emissive + occludeAmbientColor(ambient, textureChannels);

    if (material.outputMode == OUTPUT_MODE_FINAL) {
        outColor = vec4(color, alpha);
    } else if (material.outputMode == OUTPUT_MODE_COLOR) {
        outColor = vec4(baseColor.rgb, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_EMISSIVE) {
        outColor = vec4(emissive, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_METALLIC) {
        outColor = vec4(vec3(metallic), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_SPECULAR) {
        outColor = vec4(specular, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_ROUGHNESS) {
        outColor = vec4(vec3(roughness), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_OCCLUSION) {
        outColor = vec4(occludeAmbientColor(vec3(1.0), textureChannels), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_NORMAL) {
        outColor = vec4(n*0.5 + 0.5, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_ALPHA) {
        outColor = vec4(vec3(baseColor.a), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_UVS0) {
        outColor = vec4(vec2(oTexcoords0), 0.0, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_UVS1) {
        outColor = vec4(vec2(oTexcoords1), 0.0, 1.0);
    } else if (material.outputMode == OUTPUT_MODE_SSAO) {
        float ao = sampleAOMap();
        outColor = vec4(vec3(ao), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_CLEARCOAT_FACTOR) {
        outColor = vec4(vec3(clearcoatFactor), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_CLEARCOAT_ROUGHNESS) {
        outColor = vec4(vec3(clearcoatRoughness), 1.0);
    } else if (material.outputMode == OUTPUT_MODE_CLEARCOAT_NORMAL) {
        outColor = outColor = vec4(clearcoatNormal*0.5 + 0.5, 1.0);
    }
}
