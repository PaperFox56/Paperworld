#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

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

// --- CONSTANTS / SHARED CACHE ---
const uint SHARED_CACHE_SIZE = 256u; // tune this to GPU limits (48KB/96KB total shared mem matters)
shared OctreeNode shared_octree_cache[SHARED_CACHE_SIZE];

// compute nodes count for first L levels: nodes = 1 + 8 + 8^2 + ... = (8^(L+1)-1)/7
uint nodes_for_levels(uint levels) {
    uint count = 0u;
    uint stride = 1u;
    for (uint i = 0u; i <= levels; ++i) {
        count += stride;
        // watch overflow but levels will be small (<= ~8)
        stride *= 8u;
    }
    return count;
}

// Prefetch start_index..start_index+count-1 (clamped to SHARED_CACHE_SIZE and voxels.node_count).
// NOTE: the caller should compute 'count' (see main).
void prefetch_octree(uint start_index, uint count) {
    // clamp
    uint clamped = count;
    if (clamped > SHARED_CACHE_SIZE) clamped = SHARED_CACHE_SIZE;
    if (clamped > voxels.node_count) clamped = voxels.node_count;

    uint local_id = gl_LocalInvocationIndex;
    uint stride = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = local_id; i < clamped; i += stride) {
        // safe read from SSBO
        shared_octree_cache[i] = voxels.tree[start_index + i];
    }
    barrier(); // wait for all loads
}

// Return a node from cache if possible, else from SSBO
OctreeNode getNode(uint index, uint prefetch_count) {
    OctreeNode n;
    if (index < 0u || index >= voxels.node_count) {
        // return an empty PURE_LEAF
        n.node_type = PURE_LEAF;
        for (int i = 0; i < 8; ++i) {
            n.children[i] = 0u;
        }
    } else if (index < prefetch_count && index < SHARED_CACHE_SIZE) {
        n = shared_octree_cache[index];
    } else {
        n = voxels.tree[index];
    }
    return n;
}


IntersectionData rayTraceThroughVoxelGrid(RayData ray, OctreeNode grid, float size);
IntersectionData rayTraceThroughVoxelOctree(RayData ray, uint prefetch_nodes);

void main() {
    \
    // number of octree levels (grid_size is a power of two)
    int total_levels = int(floor(log2(float(voxels.grid_size))));
    // choose how many levels to prefetch
    const uint PREFETCH_LEVELS = 2u; // tune this for performance/memory tradeoff
    uint prefetch_nodes = nodes_for_levels(min(uint(total_levels), PREFETCH_LEVELS));
    // clamp to available nodes
    prefetch_nodes = min(prefetch_nodes, voxels.node_count);
    // call prefetch once per workgroup
    prefetch_octree(0u, prefetch_nodes);


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
    vec3 eps_vec = vec3(global.epsilon);
    ray_dir = normalize(sign(ray_dir) * max(abs(ray_dir), eps_vec));

    // Calculate inverse direction safely
    vec3 inv_dir = 1.0 / ray_dir;

    RayData ray = {
        ray_origin, ray_dir, inv_dir
    };

    IntersectionData inter = rayTraceThroughVoxelOctree(ray, prefetch_nodes);

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
*/IntersectionData rayTraceThroughVoxelGrid(RayData ray, OctreeNode grid, float size) {
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
        // treat the entire grid as a single voxel (speed)
        size *= 2.0;
        upper_bound = 1.0;
    }

    grid_pos /= size;

    while (grid_pos.x > 0.0 && grid_pos.y > 0.0 && grid_pos.z > 0.0
        && grid_pos.x < upper_bound && grid_pos.y < upper_bound && grid_pos.z < upper_bound) {

        step_count++;

        // positive, safe to convert to ivec3 via truncation
        ivec3 voxel_coords = ivec3(grid_pos);
        vec3 pos_relative_to_voxel = grid_pos - vec3(voxel_coords);

        // child index (x<<2) + (y<<1) + z
        uint voxel_index = (uint(voxel_coords.x) << 2) | (uint(voxel_coords.y) << 1) | uint(voxel_coords.z);

        if (voxel_index < 8u && grid.children[voxel_index] == 1u) {
            inter.debug_info.y = 1.0;

            // Find which axis is closest to 0 or 1 (we're axis aligned)
            // distances to 0 and to 1
            vec3 d0 = pos_relative_to_voxel;
            vec3 d1 = vec3(1.0) - pos_relative_to_voxel;

            // choose min distance and axis
            float m0 = min(min(d0.x, d0.y), d0.z);
            float m1 = min(min(d1.x, d1.y), d1.z);

            bool hit_at_zero = (m0 <= m1);
            int hit_axis;
            if (hit_at_zero) {
                if (m0 == d0.x) hit_axis = 0;
                else if (m0 == d0.y) hit_axis = 1;
                else hit_axis = 2;
            } else {
                if (m1 == d1.x) hit_axis = 0;
                else if (m1 == d1.y) hit_axis = 1;
                else hit_axis = 2;
            }

            // axis-aligned normal
            vec3 normal = vec3(0.0);
            normal[hit_axis] = hit_at_zero ? -1.0 : 1.0;
            inter.normal = normal;

            pos_relative_to_voxel *= size;
            pos_relative_to_voxel = mod(pos_relative_to_voxel, 1.0);

            // Compute UV by selecting the two coordinates orthogonal to the normal
            vec2 uv_local;
            if (hit_axis == 0) { // normal on X -> use z,y or y,z depending consistent ordering
                uv_local = vec2(pos_relative_to_voxel.z, pos_relative_to_voxel.y);
            } else if (hit_axis == 1) { // normal on Y -> use x,z
                uv_local = vec2(pos_relative_to_voxel.x, pos_relative_to_voxel.z);
            } else { // Z -> use x,y
                uv_local = vec2(pos_relative_to_voxel.x, pos_relative_to_voxel.y);
            }


            inter.uv = abs(uv_local);

            inter.touched = true;
            break;
        }

        // Compute entry/exit times to nearest voxel face (per-voxel T slab)
        // Everything sized relative to a voxel
        vec3 t0 = (-pos_relative_to_voxel) * ray.invDir;
        vec3 t1 = (vec3(1.0) - pos_relative_to_voxel) * ray.invDir;

        // When dir < 0, swap t0/t1 per component. Use mix/select to avoid branch.
        bvec3 neg = lessThan(ray.dir, vec3(0.0));
        vec3 tmin = mix(t0, t1, neg); // if dir<0 use t1 as min
        vec3 tmax = mix(t1, t0, neg);

        // If dir is near zero, set to big negative so it won't be chosen
        // but we used safe invDir earlier, so this is okay.

        float lower = max(max(tmin.x, tmin.y), tmin.z);
        float higher = min(min(tmax.x, tmax.y), tmax.z);

        // Advance to next voxel
        grid_pos = grid_pos + (higher + global.ray_offset) * ray.dir;
    }

    inter.grid_pos = grid_pos * size;
    inter.debug_info.x = float(step_count) * (1.0 / 3.0);

    return inter;
}


IntersectionData rayTraceThroughVoxelOctree(RayData ray, uint prefetch_nodes) {
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

    /// Calculate intersection with the bounds of the voxel grid
    vec3 t0 = -grid_pos * ray.invDir;
    vec3 t1 = (vec3(float(grid_size_u)) - grid_pos) * ray.invDir;
    bvec3 negDir = lessThan(ray.dir, vec3(0.0));    // some math trickery to avoid branching
    vec3 tmin = mix(t0, t1, negDir);
    vec3 tmax = mix(t1, t0, negDir);
    float lower_bound = max(max(tmin.x, tmin.y), tmin.z);
    float higher_bound = min(min(tmax.x, tmax.y), tmax.z);

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
    uint stack_idx[max_level];
    ivec3 node_coord_stack[max_level];
    stack_idx[0] = 0u; // root index is 0
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

        // fetch current node on-demand (cached)
        OctreeNode cur_node = getNode(stack_idx[level], prefetch_nodes);

        if (cur_node.node_type == NON_LEAF) {
            int i = (child_coordinate.x << 2) + (child_coordinate.y << 1) + child_coordinate.z;
            uint childIndex = cur_node.children[i];

            level++;
            node_coord_stack[level] = origin + child_coordinate * size;
            size /= 2;

            if (childIndex == 0u || childIndex >= voxels.node_count) {
                // simulate an empty PURE_LEAF in-place by setting stack index to an invalid marker
                // we'll set node_type directly when used by getNode fallback
                stack_idx[level] = uint(-1); // special marker -> treated as empty leaf
            } else {
                stack_idx[level] = childIndex;
            }
        } else {
            // leaf branch unchanged, but when building new_ray we still pass origin

            RayData new_ray;
            new_ray.dir = ray.dir;
            new_ray.invDir = ray.invDir;
            new_ray.pos = (grid_pos - vec3(origin));

            inter = rayTraceThroughVoxelGrid(new_ray, cur_node, size);
            grid_pos = inter.grid_pos + vec3(origin);

            if (!inter.touched) {
                size *= 2;
                level--;
            } else {
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
