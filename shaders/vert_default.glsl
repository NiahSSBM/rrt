#version 460

struct MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 tex_coords;

layout(binding = 0) readonly buffer vInput {
    MVPBuffer mvp;
    float time;
};

void main() {
    gl_Position = mvp.proj * mvp.view *  mvp.model * vec4(position, 1.0);
    tex_coords = position.xy;

    vec3 sun_position = vec3(sin(time), 1.0, cos(time));

    fragColor = color * dot(normal, sun_position);
}