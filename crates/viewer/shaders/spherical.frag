#version 450

layout(location = 0) in vec3 oPositions;

layout(binding = 0) uniform sampler2D textureSampler;

layout(location = 0) out vec4 outColor;

vec2 invATan = vec2(0.1591, 0.3183);
vec2 sampleShericalMap(vec3 position) {
    return 0.5 + (vec2(atan(position.z, position.x), asin(-position.y)) * invATan);
}

void main() {
    vec2 uv = sampleShericalMap(normalize(oPositions));
    vec3 color = texture(textureSampler, uv).rgb;
	outColor = vec4(color, 1.0);
}