#version 460

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 f_color;
layout(binding = 0) buffer colorOffset {
    float offset;
};

void main() {
    f_color = vec4(fragColor + offset);
}