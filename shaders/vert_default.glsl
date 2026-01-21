#version 460

layout(location = 0) in vec2 position;
layout(location = 0) out vec3 fragColor;
layout(binding = 0) buffer vColor {
    vec3 colors[3];
} vColors;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragColor = vColors.colors[gl_VertexIndex];
}