#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba32f) uniform image2D color_buffer;

// general parameters
layout(set = 0, binding = 1, std430) readonly buffer parameters {
    float time;
}
global;

// Camera data
layout(set = 0, binding = 2, std140) uniform Camera {
    vec3 position;
    vec3 front;
    vec3 right;
    vec3 up;
    float fov;
}
camera;

// The code we want to execute in each invocation
void main() {

    /* Let's start by calculating the direction of the ray */
    float fov = radians(camera.fov); // vertical FOV

    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec2 resolution = vec2(imageSize(color_buffer));

    vec2 uv = (vec2(pixel.x, resolution.y - 1.0 - pixel.y) + 0.5) / resolution; // flip Y
    uv = uv * 2.0 - 1.0;

    // aspect ratio
    uv.x *= resolution.x / resolution.y;

    // compute ray direction
    float scale = tan(radians(camera.fov) * 0.5);
    vec3 ray_dir = normalize(
        -camera.front + uv.x * scale * camera.right + uv.y * scale * camera.up
    );
    // Ray origin
    vec3 ray_origin = camera.position;


    // -----

    vec3 color = vec3(0., 0. ,0.);

    /// ----- Test -- Raymarching a sphere -----

    bool touched = false;
    float t = 0;

    vec3 pos = ray_origin;

    for (int i = 0; i < 100; i++) {
        float sdf = length(pos) - 1;

        if (sdf <= 0.01) {
            touched = true;
            break;
        }

        t += sdf;

        pos = ray_origin + ray_dir * t;
    }

    if (touched) {
        color = pos;
    }

    /// ----------------------------------------    

    imageStore(color_buffer, pixel, vec4(color, 1.));
}