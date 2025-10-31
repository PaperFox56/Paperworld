#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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
    float max_steps;
    int debug_mode; // replaced bool with int for safer host packing (0/1)
}
global;

// Camera data - std140 requires vec3 to occupy 16 bytes; use vec4 to be explicit
layout(set = 1, binding = 1, std140) uniform Camera {
    vec4 position; // .w unused (padding)
    vec4 front;
    vec4 right;
    vec4 up;
    vec4 cam_params; // cam_params.x = fov (degrees), other components free/padding
}
camera;


#define PURE_LEAF 0 // all children cells have the same value
#define HETEROGENIOUS_LEAF 1 // children cells have differents values
#define NON_LEAF 2
struct OctreeNode {
    uint node_type;
    uint children[8];
    //uint padding_for_gpu[3];
};

layout(set = 1, binding = 2, std430) readonly buffer VoxelData {
    float voxels_per_unit;
    // We use an octree so the grid is always cubic
    uint grid_size;
    uint node_count;
    OctreeNode tree[];
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
    vec3 grid_pos;
    vec3 normal;                // the face intersected
    vec2 uv;                    // the position on the face
    vec3 debug_info;
};

IntersectionData rayTraceThroughVoxelGrid(RayData ray, OctreeNode grid);
IntersectionData rayTraceThroughVoxelOctree(RayData ray);

void main() {
    /* Let's start by calculating the direction of the ray */
    float fov_rad = radians(camera.cam_params.x); // vertical FOV in radians

    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec2 resolution = vec2(imageSize(color_buffer));

    vec2 uv = (vec2(pixel.x, resolution.y - 1.0 - pixel.y) + 0.5) / resolution; // flip Y
    uv = uv * 2.0 - 1.0;

    // aspect ratio
    uv.x *= resolution.x / resolution.y;

    // compute ray direction
    float scale = tan(fov_rad * 0.5);
    vec3 cam_front = camera.front.xyz;
    vec3 cam_right = camera.right.xyz;
    vec3 cam_up = camera.up.xyz;

    vec3 ray_dir = normalize(
        -cam_front + uv.x * scale * cam_right + uv.y * scale * cam_up);
    // Ray origin
    vec3 ray_origin = camera.position.xyz;

    // -----

    vec4 color = vec4(0, 0, 0, 1);
    bool grid_point = false;

    // used in debug mode to do Z-buffering
    float grid_depth = 10000.;

    if (global.debug_mode != 0) {
        // Visualize a grid on the XZ plane at Y=0
        // only the lines are visible
        if (abs(ray_dir.y) > global.epsilon){
            float t = -ray_origin.y / ray_dir.y;
            if (t > 0.0) {

                grid_depth = t;

                vec3 intersect_point = ray_origin + t * ray_dir;

                float epsilon = .1 - exp(-t);

                float grid_size = 3.0; // size of each grid cell
                float x = mod(intersect_point.x, grid_size);
                float z = mod(intersect_point.z, grid_size);
                if (x < epsilon || z < epsilon) {
                    grid_point = true;

                    color = vec4(vec3(.7), 1.);
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

    IntersectionData inter = rayTraceThroughVoxelOctree(ray);

    vec4 background = vec4(0, .5, .8, 1.);
    vec4 light_color = vec4(.8, .8, .5, 1.0);

    float depth = length(inter.global_pos - ray_origin);

    if (inter.touched) {
        if (global.debug_mode != 0 && depth > grid_depth && grid_point) {
        } else {
            //color = vec4(floor(inter.grid_pos)/float(voxels.grid_size), 1.);
            color = vec4(1.);
        }
    } else {
        if (global.debug_mode != 0 && grid_point) {
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

    imageStore(color_buffer, pixel, color);
    imageStore(normal_buffer, pixel, vec4(inter.normal, 1.));
    imageStore(uv_buffer, pixel, vec4(inter.uv, 0., 1.));
    imageStore(extra_buffer, pixel, vec4(inter.debug_info, 0.));
}

bool are_close(float a, float b) {
    return abs(a - b) <= global.epsilon;
}

/*
Do a raytracing in a 2x2x2 voxel grid.
This function assumes that the ray data is normalised to the voxel space (1 voxel for 1 unit)
The ray's origin should be the intersection point of the ray with the grid.
*/
IntersectionData rayTraceThroughVoxelGrid(RayData ray, OctreeNode grid, float size) {
    IntersectionData inter;
    inter.touched = false;
    inter.normal = vec3(0.0);
    inter.uv = vec2(0.0);
    inter.debug_info = vec3(0.0);

    // Get the starting position of the ray relative to the voxel grid
    vec3 grid_pos = ray.pos;
    uint step_count = 0u;

    // define the size of the voxel grid
    float upper_bound = 2.0;

    if (grid.node_type == PURE_LEAF) {
        // We do a little trick that treat the entire grid as a single voxel, speeding up the traversal
        size *= 2;
        upper_bound = 1.0;
    }

    grid_pos /= size;

    // Now we enter a loop, we check if the current voxel is on, if so we stop, if not, we go to the next cell
    while (grid_pos.x > 0.0 && grid_pos.y > 0.0 && grid_pos.z > 0.0
        && grid_pos.x < upper_bound && grid_pos.y < upper_bound && grid_pos.z < upper_bound) {
        step_count += 1u;
        // get voxel index
        ivec3 voxel_coords = ivec3(floor(grid_pos));

        vec3 pos_relative_to_voxel = grid_pos - vec3(voxel_coords);

        // i = (x<<2) + (y<<1) + z
        uint voxel_index = (uint(voxel_coords.x) << 2) + (uint(voxel_coords.y) << 1) + uint(voxel_coords.z);

        if (voxel_index < 8u && grid.children[voxel_index] == 1u) {
            inter.debug_info.y = 1.;
            vec3 normal = vec3(0.0);
            vec3 u = vec3(0.0);
            vec3 v = vec3(0.0);

            // Find the face that was hit by checking which coordinate is closest to 0 or 1
            vec3 distances_to_0 = pos_relative_to_voxel;
            vec3 distances_to_1 = vec3(1.0) - pos_relative_to_voxel;

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
            if (hit_axis >= 0) {
                normal[hit_axis] = hit_at_zero ? -1.0 : 1.0;
            }

            // Calculate a stable tangent/bitangent basis
            vec3 n = normal;
            if (length(n) == 0.0) {
                // degenerate: set a fallback normal (should not happen when voxel is filled)
                n = vec3(0.0, 1.0, 0.0);
            }
            vec3 tangent;
            if (abs(n.x) > 0.5) {
                tangent = normalize(vec3(n.y, -n.x, 0.0));
            } else {
                tangent = normalize(vec3(0.0, n.z, -n.y));
            }
            vec3 bitangent = normalize(cross(n, tangent));
            u = tangent;
            v = bitangent;

            inter.touched = true;
            inter.normal = normal;

            pos_relative_to_voxel *= size; // back to voxel space
            pos_relative_to_voxel = mod(pos_relative_to_voxel, 1.); // local to the voxel

            inter.uv = abs(vec2(dot(u, pos_relative_to_voxel), dot(v, pos_relative_to_voxel)));

            if (inter.normal == vec3(0.0))
                inter.uv = vec2(1.0);
            break;
        }

        // we didn't get an intersection now try to reach the next cell
        // we need the side through which the ray will exit the voxel

        // here, everthing is sized relative to a voxel and we know we are already in

        // faces: x = 0, y = 0, z = 0; left side of the inequality
        vec3 t0 = -pos_relative_to_voxel * ray.invDir;
        // faces: x = Dx, y = Dy, z = Dz; right side of the inequality
        vec3 t1 = (vec3(1.0) - pos_relative_to_voxel) * ray.invDir;

        for (int i = 0; i < 3; i++) {
            if (ray.dir[i] < 0.0) {
                float temp = t0[i];
                t0[i] = t1[i];
                t1[i] = temp;
            }
            else if (are_close(ray.dir[i], 0.0)) {
                t0[i] = t1[i] = -1e30;
            }
        }

        float lower_bound = max(max(t0.x, t0.y), t0.z);
        float higher_bound = min(min(t1.x, t1.y), t1.z);

        // Advance to next voxel with proper offset to avoid precision issues
        grid_pos = grid_pos + (higher_bound + global.ray_offset) * ray.dir;
    }


    inter.grid_pos = grid_pos *= size;
    inter.debug_info.x = float(step_count)/3.;

    return inter;
}


IntersectionData rayTraceThroughVoxelOctree(RayData ray) {
    IntersectionData inter;
    inter.touched = false;

    inter.debug_info = vec3(0.0, 0.0, 0.0);

    float voxels_per_unit = voxels.voxels_per_unit;
    float voxel_size = 1.0 / voxels_per_unit;
    uint grid_size_u = voxels.grid_size;

    // Convert to float grid pos; use float conversion and half-size properly
    vec3 center_offset = vec3(float(voxels.grid_size) * 0.5);
    vec3 grid_pos = ray.pos * voxels_per_unit + center_offset;

    bool in_bounds = true;

    // faces: x = 0, y = 0, z = 0; left side of the inequality
    vec3 t0 = -grid_pos * ray.invDir;
    // faces: x = Dx, y = Dy, z = Dz; right side of the inequality
    vec3 t1 = (vec3(float(grid_size_u)) - grid_pos) * ray.invDir;

    for (int i = 0; i < 3; i++) {
        if (ray.dir[i] < 0.0) {
            float temp = t0[i];
            t0[i] = t1[i];
            t1[i] = temp;
        }
    }

    float lower_bound = max(max(t0.x, t0.y), t0.z);
    float higher_bound = min(min(t1.x, t1.y), t1.z);

    if (lower_bound >= higher_bound || higher_bound <= 0.0) {
        in_bounds = false;
    }
    else {
        grid_pos = grid_pos + (max(0.0, lower_bound) + global.ray_offset) * ray.dir;
    }
    //check alignment
    // if (voxels.tree[5].children[3] != 0x0   ) {
    //     inter.debug_info.y = 1.0; // indicate error
    //     return inter;
    // }

    if (!in_bounds) {
        return inter;
    }

   // debug, check the behaviour of `rayTraceThroughVoxelGrid`
    // RayData new_ray;
    // new_ray.dir = ray.dir;
    // new_ray.invDir = ray.invDir;
    // new_ray.pos = grid_pos;

    // inter = rayTraceThroughVoxelGrid(new_ray, OctreeNode(PURE_LEAF, uint[8](0u,0u, 0u,0u,0u,0u,0u,0u)), float(grid_size_u)/2.);
    // inter.grid_pos;

    // return inter;
    //


    // Use global.max_steps if available
    int max_steps = int(global.max_steps);
    if (max_steps <= 0) max_steps = 10;
    int _step = 0;

    /* Now we can traverse the octree and see if we get an intersection */
    // some initialisations
    const int max_level = 16;  // seriously good luck having more than 2^16 voxels per axis
    int level = 0;     // the root
    int size = int(grid_size_u) / 2; // integer size

    // This stack will keep track of the parent nodes in order to reduce the number of buffer lookup
    OctreeNode stack[max_level];
    ivec3 node_coord_stack[max_level];
    stack[0] = voxels.tree[0];
    node_coord_stack[0] = ivec3(0, 0, 0);

    bool debug = false;

    // Loop traversal - use signed level to avoid unsigned underflow
    while (level >= 0 && _step < max_steps && !inter.touched) {
        _step++;
        // determine which child we are going to
        float size_f = float(size);
        ivec3 origin = node_coord_stack[level];
        vec3 relative_pos = grid_pos - vec3(origin);
        ivec3 child_coordinate = ivec3(floor(relative_pos / size_f));

        if (child_coordinate.x < 0 || child_coordinate.y < 0 || child_coordinate.z < 0 ||
            child_coordinate.x > 1 || child_coordinate.y > 1 || child_coordinate.z > 1) {
            // out of bounds, pop up
            size *= 2;
            level--;
            if (level < 0) {
                //debug = true;
            }
            continue;
        }

        if (stack[level].node_type == NON_LEAF) {
            int i = (child_coordinate.x << 2) + (child_coordinate.y << 1) + child_coordinate.z;
            // fetch child index and bounds-check before using it
            uint childIndex = stack[level].children[i];

            level++;
            node_coord_stack[level] = origin + child_coordinate * size;
            size /= 2;
            
            if (childIndex == 0u || childIndex >= voxels.node_count) {
                // treat as empty -> create a empty leaf that will be proccessed next iteration
                stack[level] = OctreeNode(PURE_LEAF, uint[8](0u,0u,0u,0u,0u,0u,0u,0u));
            }
            else {
                // push the child node to the stack
                stack[level] = voxels.tree[childIndex];
            }

        } else {
            // node is a leaf (PURE_LEAF or HETEROGENIOUS_LEAF)
            RayData new_ray;
            new_ray.dir = ray.dir;
            new_ray.invDir = ray.invDir;
            new_ray.pos = (grid_pos - vec3(origin));

            // if (stack[level].node_type == PURE_LEAF) {
            //      if (origin ==  ivec3(0, 8, 0))
            //         debug = true;
            // }

            inter = rayTraceThroughVoxelGrid(new_ray, stack[level], size);
            grid_pos = inter.grid_pos + vec3(origin);
                
            if (!inter.touched) {
                // pop up
                size *= 2;
                level--;
            } else {
                // we have a hit, exit
                debug = true;
            }
        }
    }
    inter.grid_pos = grid_pos;
    inter.global_pos = (grid_pos - center_offset) / voxels_per_unit;

    // debug info
    inter.debug_info.z = float(_step) / float(30);
    inter.debug_info.y = debug ? 1.0 : 0.;
    if (_step > 50) {
        inter.debug_info.x = 1.;
    }

    //inter.debug_info = grid_pos + vec3(origin);

    return inter;
}
