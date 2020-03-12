#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (constant_id = 0) const uint SSAO_KERNEL_SIZE = 32;
layout (constant_id = 1) const float SSAO_RADIUS = 0.2;
layout (constant_id = 2) const float SSAO_STRENGTH = 1.0;
layout (constant_id = 3) const uint NOISE_TEXTURE_SIZE = 8;
layout (constant_id = 4) const uint SSAO_WIDTH = 1024;
layout (constant_id = 5) const uint SSAO_HEIGHT = 768;

layout(location = 0) in vec2 oCoords;
layout(location = 1) in vec3 oViewRay;

layout(binding = 0, set = 0) uniform sampler2D normalsSampler;
layout(binding = 1, set = 0) uniform sampler2D depthSampler;
layout(binding = 2, set = 0) uniform sampler2D noiseSampler;

layout(binding = 3, set = 1) uniform SSAOKernel {
	vec4 samples[SSAO_KERNEL_SIZE];
} ssaoKernel;

layout(binding = 4, set = 2) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 invertedProj;
    vec4 eye;
    float zNear;
    float zFar;
} cameraUBO;

layout(location = 0) out vec4 finalColor;

float linearDepth(vec2 uv) {
    float near = cameraUBO.zNear;
    float far = cameraUBO.zFar;
    float depth = texture(depthSampler, uv).r;
    return (near * far) / (far + depth * (near - far));
}

void main() {
    // View-space position
    vec3 position = oViewRay * linearDepth(oCoords);

    // View-space normal
    vec3 normal = normalize(texture(normalsSampler, oCoords).xyz * 2.0 - 1.0);

    // View space random vector
    vec2 noiseScale = vec2(float(SSAO_WIDTH) / float(NOISE_TEXTURE_SIZE), float(SSAO_HEIGHT) / float(NOISE_TEXTURE_SIZE));
    vec3 randomVec = texture(noiseSampler, oCoords * noiseScale).xyz;

    // View space TBN matrix
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    // Occlusion computation
    float occlusion = 0.0;
    const float bias = 0.01f;
    for (int i = 0; i < SSAO_KERNEL_SIZE; i++) {
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
    
    finalColor = vec4(vec3(pow(occlusion, SSAO_STRENGTH)), 1.0);
}
