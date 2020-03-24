#version 450
#extension GL_ARB_separate_shader_objects : enable

const uint NO_TEXTURE_ID = 255;
const uint ALPHA_MODE_MASK = 1;

// -- Inputs --
layout(location = 0) in vec3 oViewSpaceNormal;
layout(location = 1) in vec2 oTexcoords0;
layout(location = 2) in vec2 oTexcoords1;
layout(location = 3) in float oAlpha;

// -- Push constants
layout(push_constant) uniform MaterialUniform {
    float alpha;
    uint colorTextureChannel;
    uint alphaMode;
    float alphaCutoff;
} material;

// -- Samplers --
layout(binding = 3, set = 1) uniform sampler2D colorSampler;

// -- Output --
layout(location = 0) out vec4 outNormals;

vec2 getUV(uint texChannel) {
    if (texChannel == 0) {
        return oTexcoords0;
    }
    return oTexcoords1;
}

float getAlpha(uint textureChannel) {
    float alpha = material.alpha;
    if(textureChannel != NO_TEXTURE_ID) {
        vec2 uv = getUV(textureChannel);
        float sampledAlpha = texture(colorSampler, uv).a;
        alpha *= sampledAlpha;
    }
    return alpha * oAlpha;
}

bool isMasked(float alpha) {
    return material.alphaMode == ALPHA_MODE_MASK && alpha < material.alphaCutoff;
}

void main() {
    float alpha = getAlpha(material.colorTextureChannel);
    if (isMasked(alpha)) {
        discard;
    }

    outNormals = vec4((normalize(oViewSpaceNormal) * 0.5) + 0.5, 0.0);
}
