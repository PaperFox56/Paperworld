
#include <stdio.h>

#define GRID_SIZE 64
#define VOXEL_COUNT GRID_SIZE * GRID_SIZE * GRID_SIZE

/**
 This scriot essentially generate a voxel grid filled with a sphere of given radius
 */


int main(int args, char **argv) {

    const int RADIUS = 1;

    unsigned int grid[VOXEL_COUNT] = {0};

    unsigned long index = 0;

    int grid_radius = GRID_SIZE/2;

    for (int i = -grid_radius; i < grid_radius; i++) {
    for (int j = -grid_radius; j < grid_radius; j++) {
    for (int k = -grid_radius; k < grid_radius; k++) {

        if (i >= -RADIUS && i < RADIUS && j > -RADIUS && j < RADIUS && k > -RADIUS && k < RADIUS) {
            grid[index] = 1;
        } else {
            grid[index] = 0;
        }

        index++;
    }}}


    FILE *file = fopen("voxel_cube.vox", "wb");

    // we add padding because of the way the GPU pack data
    unsigned int grid_size[] = {GRID_SIZE, 0, 0, 0};

    // 16 bytes are reserved for header
    char header[16] = "RAW";

    fwrite(header, sizeof(char), sizeof(header), file);
    fwrite(grid_size, sizeof(unsigned int), 4, file);
    int r = fwrite(grid, sizeof(unsigned int), VOXEL_COUNT, file);

    fclose(file);
    printf("%x\n", r);

    return 0;
}
