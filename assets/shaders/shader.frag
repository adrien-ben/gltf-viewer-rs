#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 oNormals;

layout(location = 0) out vec4 outColor;

const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const vec3 LIGHT_DIR = vec3(1.0, -1.0, -1.0);
const vec3 OBJECT_COLOR = vec3(1.0, 1.0, 1.0);
const vec3 AMBIENT_COLOR = vec3(0.1, 0.1, 0.1);

void main() {
    vec3 color = max(dot(normalize(oNormals), -normalize(LIGHT_DIR)), 0.0) * OBJECT_COLOR + AMBIENT_COLOR * OBJECT_COLOR;
    outColor = vec4(color, 1.0);
}
