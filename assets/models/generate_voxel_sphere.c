#include <math.h>
#include <stdio.h>

#define GRID_SIZE 16
#define VOXEL_COUNT GRID_SIZE * GRID_SIZE * GRID_SIZE
/**
 This scriot essentially generate a voxel grid filled with a sphere of given radius
 */


int main(int args, char **argv) {

    const unsigned int RADIUS = 8;

    unsigned int grid[VOXEL_COUNT] = {0};

    unsigned long index = 0;

    int grid_radius = GRID_SIZE/2;

    for (int i = -grid_radius; i < grid_radius; i++) {
    for (int j = -grid_radius; j < grid_radius; j++) {
    for (int k = -grid_radius; k < grid_radius; k++) {

        if (sqrtf(i*i + j*j + k*k) <= RADIUS) {
            grid[index] = 1;
        } else {
            grid[index] = 0;
        }

        index++;
    }}}


    FILE *file = fopen("voxel_sphere.vx", "wb");

    unsigned int grid_size[] = {GRID_SIZE, GRID_SIZE, GRID_SIZE, 0};

    fwrite(&grid_size, sizeof(unsigned int), 3, file);
    int r = fwrite(&grid, sizeof(unsigned int), VOXEL_COUNT, file);

    fclose(file);
    printf("%x", r);

    return 0;
}
