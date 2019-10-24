#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 oCoords;

layout(binding = 0) uniform sampler2D inputImage;

layout(location = 0) out vec4 finalColor;

void main() {
    vec3 color = texture(inputImage, oCoords).rgb;

    color = color/(color + 1.0);
    color = pow(color, vec3(1.0/2.2));

    finalColor = vec4(color, 1.0);
}
