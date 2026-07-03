#version 460

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(binding = 1) uniform sampler s;
layout(binding = 2) uniform texture2D tex;

void main() {
    f_color = texture(sampler2D(tex, s), tex_coords) * fragColor;
}