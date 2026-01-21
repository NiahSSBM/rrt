#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vOutput {
    vec3 out_position;
    vec4 out_color;
};

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_color = color;
}