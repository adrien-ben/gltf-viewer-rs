const uint NO_TEXTURE_ID = 255;

const uint ALPHA_MODE_MASK = 1;
const uint ALPHA_MODE_BLEND = 2;
const float ALPHA_CUTOFF_BIAS = 0.0000001;

const uint METALLIC_ROUGHNESS_WORKFLOW = 0;

struct Material {
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
    uint colorTextureChannel;
    uint materialTextureChannel;
    uint emissiveTextureChannel;
    uint normalsTextureChannel;
    uint occlusionTextureChannel;
    uint clearcoatFactorTextureChannel;
    uint clearcoatRoughnessTextureChannel;
    uint clearcoatNormalsTextureChannel;
    uint alphaMode;
    bool isUnlit;
    uint workflow;
    float ior;
    mat4 colorTextureTransform;
    mat4 materialTextureTransform;
    mat4 emissiveTextureTransform;
    mat4 normalsTextureTransform;
    mat4 occlusionTextureTransform;
    mat4 clearcoatFactorTextureTransform;
    mat4 clearcoatRoughnessTextureTransform;
    mat4 clearcoatNormalsTextureTransform;
};
