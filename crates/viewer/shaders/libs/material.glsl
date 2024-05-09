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
    uint clearcoatNormalTextureChannel;
    uint alphaMode;
    bool isUnlit;
    uint workflow;
};

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

TextureChannels getTextureChannels(Material m) {
    return TextureChannels(
        m.colorTextureChannel,
        m.materialTextureChannel,
        m.emissiveTextureChannel,
        m.normalsTextureChannel,
        m.occlusionTextureChannel,
        m.clearcoatFactorTextureChannel,
        m.clearcoatRoughnessTextureChannel,
        m.clearcoatNormalTextureChannel
    );
}
