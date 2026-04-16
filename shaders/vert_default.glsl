#version 460

struct MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 fragColor;
layout(binding = 0) readonly buffer vInput {
    MVPBuffer mvp;
};

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(position, 1.0);
    fragColor = color;
}