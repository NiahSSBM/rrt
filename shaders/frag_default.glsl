#version 460

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 f_color;
layout(binding = 1) buffer colorOffset {
    float offset;
    float dummy1;
    float dummy2;
    float dummy3;
};

void main() {
    f_color = vec4(fragColor + offset);
}