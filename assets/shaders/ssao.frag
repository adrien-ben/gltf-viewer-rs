#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (constant_id = 0) const int SSAO_KERNEL_SIZE = 64;
layout (constant_id = 1) const float SSAO_RADIUS = 0.2;

const float NEAR_CLIP = 0.01;
const float FAR_CLIP = 10.0;
const float PROJ_A = FAR_CLIP / (FAR_CLIP - NEAR_CLIP);
const float PROJ_B = (-FAR_CLIP * NEAR_CLIP) / (FAR_CLIP - NEAR_CLIP);

layout(location = 0) in vec2 oCoords;
layout(location = 1) in vec3 oViewRay;

layout(binding = 0, set = 0) uniform sampler2D normalsSampler;
layout(binding = 1, set = 0) uniform sampler2D depthSampler;
layout(binding = 2, set = 0) uniform sampler2D noiseSampler;

layout(binding = 3, set = 0) uniform SSAOKernel {
	vec4 samples[SSAO_KERNEL_SIZE];
} ssaoKernel;

layout(binding = 4, set = 1) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     mat4 invertedProj;
     vec3 eye;
} cameraUBO;

layout(location = 0) out vec4 finalColor;

float linearDepth(vec2 uv) {
    return PROJ_B / (texture(depthSampler, uv).r - PROJ_A);
}

void main() {
    // View-space position
    vec3 viewRay = normalize(oViewRay);
    vec3 position = viewRay * linearDepth(oCoords);

    // View-space normal
    vec3 normal = normalize(texture(normalsSampler, oCoords).xyz * 2.0 - 1.0);

    // View space random vector
    // TODO: compute these on the cpu and pass them as specialized constants
    ivec2 noiseSize = textureSize(noiseSampler, 0);
    ivec2 screenSize = textureSize(depthSampler, 0);
    vec2 noiseScale = vec2(float(screenSize.x) / float(noiseSize.x), float(screenSize.y) / float(noiseSize.y));
    vec3 randomVec = texture(noiseSampler, oCoords * noiseScale).xyz * 2.0 - 1.0;

    // View space TBN matrix
    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn       = mat3(tangent, bitangent, normal);

    // Occlusion computation
    float occlusion = 0.0;
    const float bias = 0.01f;
    for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
        // get sample position:
        vec3 kSample = tbn * ssaoKernel.samples[i].xyz;
        kSample = kSample * SSAO_RADIUS + position;
        
        // project sample position:
        vec4 offset = vec4(kSample, 1.0);
        offset = cameraUBO.proj * offset;
        offset.xy /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;
        
        float depth = -linearDepth(offset.xy);

        // range check & accumulate:
        float rangeCheck = smoothstep(0.0f, 1.0f, SSAO_RADIUS / abs(depth - position.z));
		occlusion += (depth >= kSample.z + bias ? 1.0f : 0.0f) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));
    
    finalColor = vec4(vec3(occlusion), 1.0);
}
