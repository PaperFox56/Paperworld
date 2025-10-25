#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

// general parameters
layout(set = 0, binding = 0, std430) readonly buffer parameters {
    float time;
}
global;

layout(set = 0, binding = 1, rgba32f) uniform image2D color_buffer;

// Camera data
// layout(set = 0, binding = 2, std430) readonly buffer Camera {
//     vec3 position;
//     vec3 front;
//     vec3 up;
//     vec3 right;
//     float fov;
// }
// camera;

// The code we want to execute in each invocation
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);

    vec4 color = vec4(uv.x/1024.0, uv.y/1024., sin(global.time), 1.);

    imageStore(color_buffer, uv, color);
}