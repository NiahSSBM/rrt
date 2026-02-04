#version 460

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 fragColor;
layout(binding = 0) buffer vColor {
    vec4 colors[3];
};

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}