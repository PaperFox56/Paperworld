#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba32f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba32f) uniform image2D normal_buffer;
layout(set = 0, binding = 2, rgba32f) uniform image2D uv_buffer;
layout(set = 0, binding = 3, rgba32f) uniform image2D extra_buffer;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 resolution = imageSize(color_buffer);

    vec4 color = imageLoad(color_buffer, pixel);
    vec4 normal = imageLoad(normal_buffer, pixel);
    vec4 uv = imageLoad(uv_buffer, pixel);
    vec4 extra = imageLoad(extra_buffer, pixel);

    vec3 position = extra.xyz;
    float depth = extra.w;

    if (color == vec4(1.)) {
        color =  vec4(uv.xyz, 1.);
    }

    //color =  vec4(extra.xyz, 1.);

    imageStore(color_buffer, pixel, color);
}