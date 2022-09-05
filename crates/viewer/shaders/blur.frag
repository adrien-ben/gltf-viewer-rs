#version 450

layout(location = 0) in vec2 oCoords;

layout(binding = 0) uniform sampler2D inputImage;

layout(location = 0) out vec4 finalColor;

void main() {
	const int blurRange = 2;
	int n = 0;
	vec2 texelSize = 1.0 / vec2(textureSize(inputImage, 0));
	float result = 0.0;
	for (int x = -blurRange; x < blurRange; x++) {
		for (int y = -blurRange; y < blurRange; y++) {
			vec2 offset = vec2(float(x), float(y)) * texelSize;
			result += texture(inputImage, oCoords + offset).r;
			n++;
		}
	}
	finalColor = vec4(vec3(result / float(n)), 1.0);
}
