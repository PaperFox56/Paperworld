// Take a vox file, check the header to see if it's raw voxel data, and convert
// it to an octree structure
#include <bits/types/stack_t.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <assert.h>

#define MAX_NODES 1000000

typedef struct OctreeNode OctreeNode;
typedef struct OctreeNodeGPU OctreeNodeGPU;

#define PURE_LEAF 0 // all children cells have the same value
#define HETEROGENIOUS_LEAF 1 // children cells have differents values
#define NON_LEAF 2


// Used for cpu side representation of the octree
struct OctreeNode {
  uint32_t value; // only valid if is_leaf is 1
  OctreeNode *children[8];
  bool is_leaf;
  bool all_children_are_leaves;
};

// Used for GPU side representation of the octree

struct  __attribute__((aligned(16))) OctreeNodeGPU {
  uint32_t node_type; // number of children (0-8)
  uint32_t children[8];
  //uint32_t padding_for_gpu[3];
};

OctreeNode *build_octree(uint32_t *voxel_data, uint32_t grid_size);

void freeNode(OctreeNode *node) {
  if (node == NULL)
    return;

  for (int i = 0; i < 8; i++) {
    freeNode(node->children[i]);
  }
  free(node);
}

void print_octree(const OctreeNode *root, int depth);
bool validate_octree(const OctreeNodeGPU *nodes, uint32_t node_count);

// Breadth-first (level order) traversal version
uint32_t fill_flat_array(OctreeNode *root,
                         OctreeNodeGPU *flat_array)
{
    if (!root) return 0;

    uint32_t node_count = 0;

    // Simple queue to process nodes level by level
    OctreeNode **queue = malloc(sizeof(OctreeNode*) * MAX_NODES);
    uint32_t *index_queue = malloc(sizeof(uint32_t) * MAX_NODES);
    assert(queue && index_queue);

    uint32_t head = 0, tail = 0;

    // enqueue root
    queue[tail] = root;
    index_queue[tail] = node_count++;
    tail++;

    while (head < tail) {
        OctreeNode *node = queue[head];
        uint32_t current_index = index_queue[head];
        head++;

        OctreeNodeGPU *gpu_node = &flat_array[current_index];

        if (node->is_leaf) {
            gpu_node->node_type = PURE_LEAF;
            for (int i = 0; i < 8; i++) {
                gpu_node->children[i] = node->value;
            }
        } else if (node->all_children_are_leaves) {
            gpu_node->node_type = HETEROGENIOUS_LEAF;
            for (int i = 0; i < 8; i++) {
                if (node->children[i])
                    gpu_node->children[i] = node->children[i]->value;
                else
                    gpu_node->children[i] = 0;
            }
        } else {
            gpu_node->node_type = NON_LEAF;
            for (int i = 0; i < 8; i++) {
                if (node->children[i]) {
                    gpu_node->children[i] = node_count;
                    queue[tail] = node->children[i];
                    index_queue[tail] = node_count;
                    tail++;
                    node_count++;
                    if (node_count >= MAX_NODES)
                        printf("Warning: node overflow (%u)\n", node_count);
                } else {
                    gpu_node->children[i] = 0; // invalid
                }
            }
        }
    }

    free(queue);
    free(index_queue);
    return node_count;
}


int main(int args, char **argv) {
  // get the file name from command line
  if (args < 2) {
    printf("Usage: %s <input_file.vx>\n", argv[0]);
    return 1;
  }

  char *input_filename = argv[1];
  FILE *file = fopen(input_filename, "rb");
  if (!file) {
    printf("Could not open file: %s\n", input_filename);
    return 1;
  }

  // read and validate header
  char header[16];
  fread(header, sizeof(char), 16, file);
  if (strncmp(header, "RAW", 3) != 0) {
    printf("Invalid file format. Expected RAW header.\n");
    fclose(file);
    return 1;
  }

  uint32_t grid_size;
  fread(&grid_size, sizeof(uint32_t), 1, file);
  // move the cursor to skip unused size values
  fseek(file, sizeof(uint32_t) * 3, SEEK_CUR);

  // The grid size must be a power of two for octree construction
  if ((grid_size & (grid_size - 1)) != 0) {
    printf("Error: Grid size must be a power of two.\n");
    fclose(file);
    return 1;
  }

  // read voxel data
  uint32_t voxel_count = grid_size * grid_size * grid_size;
  uint32_t *voxel_data = (uint32_t *)malloc(sizeof(uint32_t) * voxel_count);
  fread(voxel_data, sizeof(uint32_t), voxel_count, file);

  fclose(file);

  // get the maximun number of nodes in the octree and crash if it exceeds the
  // allowed limit
  uint32_t levels = log2f(grid_size);
  uint32_t required_nodes =
      (uint32_t)(pow(8, levels) - 1); // geometric series sum

  if (required_nodes > MAX_NODES) {
    printf("Error: Voxel grid too large to convert to octree. Requires %u "
           "nodes, but max is %u.\n",
           required_nodes, MAX_NODES);
    free(voxel_data);
    return 1;
  }
  OctreeNode *root = build_octree(voxel_data, grid_size);

  // we don't need the voxel data anymore
  free(voxel_data);

  // Now let's convert the pointer representation to a flat array for GPU
  // consumption Instead of pointers, each node will store the indices of its
  // children in the flat array

  OctreeNodeGPU *flat_array =
      (OctreeNodeGPU *)malloc(sizeof(OctreeNodeGPU) * MAX_NODES);
  uint32_t node_count = 0;

  node_count = fill_flat_array(root, flat_array);

  printf("Nodes: %d\n", node_count);

  // Finally write the flat array to a new file, if no argument is provided,
  // default to {original_filename}_octree.vox

  char output_filename[256];
  if (args >= 3) {
    strncpy(output_filename, argv[2], 256);
  } else {
    int name_without_extension = strlen(input_filename);
    input_filename[name_without_extension - 4] = 0;
    snprintf(output_filename, 256, "%s_octree.vox", input_filename);
  }

  FILE *output_file = fopen(output_filename, "wb");
  if (!output_file) {
    printf("Error while saving: could not open output file: %s\n",
           output_filename);
    free(flat_array);
    freeNode(root);
    return 1;
  }

  // header for octree file
  char output_header[16] = "OCTREE";
  int size_data[4] = {grid_size, 0, 0, 1};
  fwrite(output_header, sizeof(char), sizeof(output_header), output_file);
  fwrite(size_data, sizeof(int), 4, output_file);
  int node_count_data[4] = {node_count, 0, 0, 2};
  fwrite(node_count_data, sizeof(uint32_t), 4, output_file);
  fwrite(flat_array, sizeof(OctreeNodeGPU), node_count, output_file);

  fclose(output_file);

  // free the whole thing

  validate_octree(flat_array, node_count);

  //print_octree(root, 0);

  free(flat_array);
  freeNode(root);

  printf("Done, file saved to %s\n", output_filename);

  return 0;
}

OctreeNode *build_node(uint32_t *voxel_data, uint32_t grid_size, uint32_t x,
                       uint32_t y, uint32_t z, uint32_t level);

/*
    To build the octree, we will go from the root to the leaves.
    For each node, we will check if all its children are the same value. If they
   are, we will make the node a leaf node with that value. And get rid of the
   children. If not, we will keep the children and mark the node as non-leaf.

    The final octree will be stored in a flat array, where each node has an
   index.

    Start at level levels-1 (the root), and go down to level 0 (the leaves).
    Each node at level L covers a 2^L x 2^L x 2^L region of the voxel grid.
    For each node, we will calculate its children by subdividing its region into
   8 octants. Each octant will correspond to a child node at level L-1. Process
   the children recursively until we reach the leaves at level 0.
*/
OctreeNode *build_octree(uint32_t *voxel_data, uint32_t grid_size) {
  uint32_t levels = log2f(grid_size);
  OctreeNode *root = build_node(voxel_data, grid_size, 0, 0, 0, levels);
  return root;
}

OctreeNode *build_node(uint32_t *voxel_data, uint32_t grid_size, uint32_t x,
                       uint32_t y, uint32_t z, uint32_t level) {
  OctreeNode *node = (OctreeNode *)malloc(sizeof(OctreeNode));

  if (level == 0) {
    // Leaf node
    uint32_t index = x + y * grid_size + z * grid_size * grid_size;
    node->value = voxel_data[index];
    node->is_leaf = true;
    node->all_children_are_leaves = false;

    for (int i = 0; i < 8; i++) {
      node->children[i] = NULL;
    }

    return node;
  }

  // Non-leaf node
  uint32_t half_size = 1 << (level - 1);
  bool all_same = true;
  bool all_leaves = true;
  uint32_t first_value = 0;

  for (int i = 0; i < 8; i++) {
    // Calculate offsets for each octant
    // i = xyz in binary
    uint32_t offset_x = (i & 4) ? half_size : 0;
    uint32_t offset_y = (i & 2) ? half_size : 0;
    uint32_t offset_z = (i & 1) ? half_size : 0;

    node->children[i] = build_node(voxel_data, grid_size, x + offset_x,
                                   y + offset_y, z + offset_z, level - 1);

    if (!node->children[i]->is_leaf) {
      all_leaves = false;
      all_same = false;
    }

    if (i == 0) {
      first_value = node->children[i]->value;
    } else {
      if (node->children[i]->value != first_value) {
        all_same = false;
      }
    }
  }

  if (all_same) {
    // Make this node a leaf
    node->value = first_value;
    node->is_leaf = true;
    // Free children
    for (int i = 0; i < 8; i++) {
      free(node->children[i]);
      node->children[i] = NULL;
    }
  } else {
    node->is_leaf = false;
    node->value = 1; // there is something there

    node->all_children_are_leaves = all_leaves;

    // free empty spaces
    for (int i = 0; i < 8; i++) {
      if (node->children[i]->value == 0) {
        free(node->children[i]);
        node->children[i] = NULL;
      }
    }
  }

  return node;
}


bool validate_octree(const OctreeNodeGPU *nodes, uint32_t node_count) {
    for (uint32_t i = 0; i < node_count; i++) {
        const OctreeNodeGPU *n = &nodes[i];
        if (n->node_type == 2) { // NON_LEAF
            bool has_child = false;
            for (int j = 0; j < 8; j++) {
                uint32_t c = n->children[j];
                if (c > 0) {
                    has_child = true;
                    if (c >= node_count) {
                        printf("Invalid child index %u in node %u\n", c, i);
                        return false;
                    }
                    if (c == i) {
                        printf("Self-reference at node %u\n", i);
                        return false;
                    }
                }
            }
            if (!has_child) {
                printf("Empty non-leaf node %u\n", i);
                return false;
            }
        }
    }

    return true;
}

// debug function that print the octree, with indentation for depth
void print_octree(const OctreeNode *root, int depth) {
    if (root == NULL) {
        return;
    }

    for (int i = 0; i < depth; i++) printf("  ");

    if (root->is_leaf) {
        printf("Leaf: %u\n", root->value);
    } else {
        printf("Node, AllchildrenAreLeaves: %d\n", root->all_children_are_leaves);
        for (int i = 0; i < 8; i++) {
            print_octree(root->children[i], depth + 1);
        }
    }
}
