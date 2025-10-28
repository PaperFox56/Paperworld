#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

/// Frame buffers
layout(set = 0, binding = 0, rgba32f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba32f) uniform image2D normal_buffer;
layout(set = 0, binding = 2, rgba32f) uniform image2D uv_buffer;
layout(set = 0, binding = 3, rgba32f) uniform image2D extra_buffer;

// general parameters
layout(set = 1, binding = 0, std430) readonly buffer parameters {
    float time;
    float epsilon;
    float ray_offset;
    float intersection_offset;
    int max_steps;
    bool debug_mode;
}
global;

// Camera data
layout(set = 1, binding = 1, std140) uniform Camera {
    vec3 position;
    vec3 front;
    vec3 right;
    vec3 up;
    float fov;
}
camera;

layout(set = 1, binding = 2, std430) readonly buffer VoxelData {
    float voxels_per_unit;
    uvec3 grid_size;
    uint data[];
}
voxels;

struct RayData {
    vec3 pos;
    vec3 dir;
    vec3 invDir;    // Inverse of the direction
};

struct IntersectionData {
    bool touched;
    vec3 global_pos;
    uint voxel_index;
    ivec3 voxel_coordinates;    // position of the voxel in the grid
    vec3 normal;                // the face intersected
    vec2 uv;                    // the position on the face

    float depth;                // distance from the camera
};

IntersectionData rayTraceThroughVoxelGrid(RayData ray);

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
        -camera.front + uv.x * scale * camera.right + uv.y * scale * camera.up);
    // Ray origin
    vec3 ray_origin = camera.position;

    // -----

    vec4 color = vec4(0, 0, 0, 1);
    bool grid_point = false;

    if (global.debug_mode) {
        // Visualize a grid on the XZ plane at Y=0
        // only the lines are visible
        if (abs(ray_dir.y) > global.epsilon){
            float t = -ray_origin.y / ray_dir.y;
            if (t > 0.0) {
                float epsilon = .03;

                vec3 intersect_point = ray_origin + t * ray_dir;
                float grid_size = 3.0; // size of each grid cell
                float x = mod(intersect_point.x, grid_size);
                float z = mod(intersect_point.z, grid_size);
                if (x < epsilon || z < epsilon) {
                    grid_point = true;

                    color = vec4(1.);
                    if (intersect_point.x > -epsilon && intersect_point.x < epsilon) {
                        color = vec4(1., 0., 0., 1.);
                    }
                    else if (intersect_point.z > -epsilon && intersect_point.z < epsilon) {
                        color = vec4(0., 0., 1., 1.);
                    }
                }
            }
        }
    }

    // Handle near-zero components in ray direction
    for (int i = 0; i < 3; i++) {
        if (abs(ray_dir[i]) < global.epsilon) {
            ray_dir[i] = sign(ray_dir[i]) * global.epsilon;
        }
    }
    ray_dir = normalize(ray_dir);

    // Calculate inverse direction safely
    vec3 inv_dir;
    for (int i = 0; i < 3; i++) {
        inv_dir[i] = abs(ray_dir[i]) > global.epsilon ? 1.0 / ray_dir[i] : 1e30 * sign(ray_dir[i]);
    }

    RayData ray = {
        ray_origin, ray_dir, inv_dir
    };

    IntersectionData inter = rayTraceThroughVoxelGrid(ray);

    vec4 background = vec4(0, .5, .8, 1.);
    vec4 light_color = vec4(.8, .8, .5, 1.0);

    if (inter.touched) {
        color = vec4(1.);
    } else {
        if (global.debug_mode && grid_point) {
            //color = vec4(1.);
        } else {
            // create a gradient to simulate the sky
            
            vec3 horizon = normalize(vec3(ray.dir.x, 0.0, ray.dir.z));
            float angle_to_horizon = ray.dir.y;

            if (ray.dir.y < 0.0) {
                color = vec4(.7) + angle_to_horizon * vec4(0.2 * vec3(1.0), 1.0);
            } else {
                color = background + (1 - angle_to_horizon) * 0.5 * light_color + angle_to_horizon * vec4(0.2 * vec3(1.0), 1.0);
            }
        }

    }

    // color = vec4(voxels.grid_size/16, 1.);
    // color = vec4(1);

    imageStore(color_buffer, pixel, color);
    imageStore(normal_buffer, pixel, vec4(inter.normal, 1.));
    imageStore(uv_buffer, pixel, vec4(inter.uv, 0., 1.));
    imageStore(extra_buffer, pixel, vec4(inter.global_pos, inter.depth));
}

bool are_close(float a, float b) {
    return abs(a - b) <= global.epsilon;
}

IntersectionData rayTraceThroughVoxelGrid(RayData ray) {
    IntersectionData inter;
    inter.touched = false;

    float voxels_per_unit = voxels.voxels_per_unit;
    float voxel_size = 1. / voxels_per_unit;
    //vec3 grid_size_in_units = voxels.grid_size / voxels_per_unit;

    // Get the starting position of the ray relative to the voxel grid
    vec3 grid_pos = ray.pos * voxels_per_unit + voxels.grid_size / 2;

    bool in_bounds = true;

    // check if we are in the bounds
    /*
    * P is the position vector, R is the ray direction, D is the dimention vector of the grid
     * the ray has for expression: P = P0 + t*R
     * for every coordinate i we have: (-Pi / Ri) < t < (Di - Pi) / Ri or (-Pi / Ri) > t > (Di - Pi) depending on the sign of Ri
     *
     * So we need to calculate every boundary of the possibles intervals of t, then chose the smallest value that is still positive
     */

    // faces: x = 0, y = 0, z = 0; left side of the inaquation
    vec3 t0 = -grid_pos * ray.invDir;
    // faces: x = Dx, y = Dy, z = Dz; right side of the inaquation
    vec3 t1 = (voxels.grid_size - grid_pos) * ray.invDir;

    for (int i = 0; i < 3; i++) {
        if (ray.dir[i] < 0) {
            float temp = t0[i];
            t0[i] = t1[i];
            t1[i] = temp;
        }
    }

    float lower_bound = max(max(t0.x, t0.y), t0.z);
    float higher_bound = min(min(t1.x, t1.y), t1.z);

    if (lower_bound >= higher_bound || higher_bound <= 0) {
        in_bounds = false;
    }
    else {
        grid_pos = grid_pos + (max(0.0, lower_bound) + global.ray_offset) * ray.dir;
    }
    /*
    *    if (lower_bound > higher_bound) {
     *        first_voxel_touched = ivec3(100);
     *        in_bounds = true;
    }*/

    if (!in_bounds) {
        return inter;
    }

    int max_steps = global.max_steps;
    int _step = 0;

    // Now we enter a loop, we check if the current voxel is on, if so we stop, if not, we go to the next cell
    while (grid_pos.x >= 0 && grid_pos.y >= 0 && grid_pos.z >= 0
        && grid_pos.x < voxels.grid_size.x && grid_pos.y < voxels.grid_size.y && grid_pos.z < voxels.grid_size.z
        && _step < max_steps) {
        // get voxel index
        ivec3 voxel_coords = ivec3(grid_pos);
        vec3 pos_relative_to_voxel = grid_pos - voxel_coords;

        // i = x * H * D + y * D + z
        uint voxel_index = (voxel_coords.x * voxels.grid_size.y + voxel_coords.y) * voxels.grid_size.z + voxel_coords.z;

        if (voxels.data[voxel_index] == 1) {
            vec3 normal = vec3(0);
            vec3 u, v;

            bool negative_one = false;

            // Find the face that was hit by checking which coordinate is closest to 0 or 1
            vec3 distances_to_0 = pos_relative_to_voxel;
            vec3 distances_to_1 = vec3(1) - pos_relative_to_voxel;

            float min_dist = 1.0;
            int hit_axis = -1;
            bool hit_at_zero = true;

            for (int i = 0; i < 3; i++) {
                if (distances_to_0[i] < min_dist) {
                    min_dist = distances_to_0[i];
                    hit_axis = i;
                    hit_at_zero = true;
                }
                if (distances_to_1[i] < min_dist) {
                    min_dist = distances_to_1[i];
                    hit_axis = i;
                    hit_at_zero = false;
                }
            }

            // Set the normal based on which face was hit
            normal = vec3(0);
            if (hit_axis >= 0) {
                normal[hit_axis] = hit_at_zero ? -1 : 1;
            }

            // Calculate UV basis vectors
            for (int i = 0; i < 3; i++) {
                u[(i+1) % 3] = abs(normal[i]);
                v[(i+2) % 3] = abs(normal[i]);
            }

            inter.touched = true;
            inter.voxel_index = voxel_index;
            inter.voxel_coordinates = voxel_coords;
            inter.normal = normal;
            inter.uv = vec2(dot(u, pos_relative_to_voxel), dot(v, pos_relative_to_voxel));

            vec3 absolute_pos = grid_pos * voxel_size;
            inter.global_pos = absolute_pos;
            inter.depth = length(absolute_pos - ray.pos);

            if (normal == vec3(0))
            inter.uv = vec2(1);
            break;
        }

        // we didn't get an intersection now try to reach the next cell
        // we need the side through which the ray will exit the voxel

        // here, everthing is sized relative to a voxel and we know we are already in

        // faces: x = 0, y = 0, z = 0; left side of the inaquation
        vec3 t0 = -pos_relative_to_voxel * ray.invDir;
        // faces: x = Dx, y = Dy, z = Dz; right side of the inaquation
        vec3 t1 = (vec3(1) - pos_relative_to_voxel) * ray.invDir;

        for (int i = 0; i < 3; i++) {
            if (ray.dir[i] < 0) {
                float temp = t0[i];
                t0[i] = t1[i];
                t1[i] = temp;
            }
            else if (are_close(ray.dir[i], 0)) {
                t0[i] = t1[i] = -100000;
            }
        }

        float lower_bound = max(max(t0.x, t0.y), t0.z);
        float higher_bound = min(min(t1.x, t1.y), t1.z);

        // Advance to next voxel with proper offset to avoid precision issues
        grid_pos = grid_pos + (higher_bound + global.intersection_offset) * ray.dir;

        _step++;
    }

    return inter;
}
