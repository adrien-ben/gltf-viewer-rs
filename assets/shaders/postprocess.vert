#version 450
#extension GL_ARB_separate_shader_objects : enable

const vec4 VERTICES[6] =  vec4[](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0),
    vec4(1.0, -1.0, 1.0, 0.0),
    vec4(1.0, -1.0, 1.0, 0.0),
    vec4(-1.0, -1.0, 0.0, 0.0),
    vec4(-1.0, 1.0, 0.0, 1.0)
);

layout(location = 0) out vec2 oCoords;

void main() {
    vec4 vertex = VERTICES[gl_VertexIndex];

    oCoords = vec2(vertex.z, vertex.w);
    gl_Position = vec4(vertex.x, vertex.y, 0.0, 1.0);
}
