// https://wunkolo.github.io/post/2022/07/gl_ext_fragment_shader_barycentric-wireframe/

#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 f_color;

void main() {
    const vec3 BaryCoord = gl_BaryCoordEXT;
    const vec3 dBaryCoordX = dFdx(BaryCoord);
    const vec3 dBaryCoordY = dFdy(BaryCoord);
    const vec3 dBaryCoord = sqrt(dBaryCoordX * dBaryCoordX + dBaryCoordY * dBaryCoordY);
    const float Thickness = 1.5;
    const vec3 Remap = smoothstep(
		vec3(0.0),
		dBaryCoord * Thickness,
		BaryCoord
	);

    const float Wireframe = 1.0 - min(Remap.x, min(Remap.y, Remap.z));
    f_color = vec4(Wireframe.xxx, 1.0);
}