#version 460

struct MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(location = 0) in vec3 position;
layout(location = 0) out vec4 fragColor;
layout(binding = 0) buffer vInput {
    vec4 colors[3];
    MVPBuffer mvp;
};

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(position, 1.0);
    fragColor = colors[gl_VertexIndex % 3];
}