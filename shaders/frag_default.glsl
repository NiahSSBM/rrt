#version 460

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 f_color;
layout(binding = 1) uniform sampler s;
layout(binding = 2) uniform texture2D tex;

void main() {
    f_color = vec4(fragColor);
}