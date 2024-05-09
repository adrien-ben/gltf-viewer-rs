struct Camera {
    mat4 view;
    mat4 proj;
    mat4 invertedProj;
    vec4 eye;
    float zNear;
    float zFar;
};
