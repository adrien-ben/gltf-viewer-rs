#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPositions;

layout(push_constant) uniform Camera {
    layout(offset = 0) mat4 viewProj;
} camera;


layout(location = 0) out vec3 oPositions;

void main() {
    oPositions = vPositions;
    gl_Position = camera.viewProj * vec4(vPositions, 1.0);
    // TODO: why do I need to do that ? (and cull the front face)
    gl_Position.x *= -1;
}
