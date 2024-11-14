#define MEMORY_ARENA_IMPLEMENTATION
#include "memory_arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

// Define the Vector3 structure here
typedef struct {
    int x, y, z;
} Vector3;

#define COLOR_RESET "\033[0m"
#define COLOR_COMMENTARY "\033[33m"
#define COLOR_SUCCESS "\033[32m"
#define COLOR_ERROR "\033[31m"
#define COLOR_INFO "\033[35m"
#define COLOR_SECTION "\033[36m"

// Print memory usage in MB
void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double memory_in_mb = usage.ru_maxrss / 1024.0; // Convert from KB to MB
    printf(COLOR_INFO "Memory Usage: %.2f MB\n" COLOR_RESET, memory_in_mb);
}

int main() {
    // Start timing
    clock_t start_time = clock();

    // Initialize a large memory arena (1 MB)
    MemoryArena arena;
    MemoryArena_Init(&arena, 1024 * 1024); // 1 MB arena

    print_memory_usage();

    printf(COLOR_SECTION "Arena initialized with 1MB of memory\n" COLOR_RESET);
    MemoryArena_Debug(&arena);
    print_memory_usage();

    // Allocate a Vector3 from the arena
    Vector3 *vec = (Vector3 *)MemoryArena_Alloc(&arena, sizeof(Vector3));
    if (vec) {
        vec->x = 1;
        vec->y = 2;
        vec->z = 3;
        printf(COLOR_SUCCESS "Allocated Vector3: (%d, %d, %d)\n" COLOR_RESET, vec->x, vec->y, vec->z);
    } else {
        printf(COLOR_ERROR "Failed to allocate Vector3\n" COLOR_RESET);
    }

    // Allocate an array of integers from the arena
    int *array = (int *)MemoryArena_Alloc(&arena, sizeof(int) * 100);
    if (array) {
        for (int i = 0; i < 100; i++) {
            array[i] = i * i;
        }
        printf(COLOR_SUCCESS "Array[50]: %d\n" COLOR_RESET, array[50]);
    } else {
        printf(COLOR_ERROR "Failed to allocate integer array\n" COLOR_RESET);
    }

    // Perform aligned allocation (alignment of 64 bytes)
    void *aligned_memory = MemoryArena_AllocAligned(&arena, 256, 64);
    if (aligned_memory) {
        printf(COLOR_SUCCESS "Allocated 256 bytes of aligned memory (aligned to 64 bytes)\n" COLOR_RESET);
    } else {
        printf(COLOR_ERROR "Failed to allocate aligned memory\n" COLOR_RESET);
    }

    // Duplicate a string into the memory arena
    const char *message = "Hello, Memory Arena!";
    char *dup_str = MemoryArena_StrDup(&arena, message);
    if (dup_str) {
        printf(COLOR_SUCCESS "Duplicated string: %s\n" COLOR_RESET, dup_str);
    } else {
        printf(COLOR_ERROR "Failed to duplicate string\n" COLOR_RESET);
    }

    // Reset the arena for reuse
    MemoryArena_Reset(&arena);
    printf(COLOR_SECTION "MemoryArena Reset\n" COLOR_RESET);
    MemoryArena_Debug(&arena);

    // Allocate again after reset
    char *reset_message = (char *)MemoryArena_Alloc(&arena, 256);
    if (reset_message) {
        strcpy(reset_message, "This message is after reset.");
        printf(COLOR_SUCCESS "%s\n" COLOR_RESET, reset_message);
    } else {
        printf(COLOR_ERROR "Failed to allocate memory after reset\n" COLOR_RESET);
    }

    MemoryArena_Debug(&arena);
    print_memory_usage();

    // Free the memory arena
    MemoryArena_Free(&arena);
    printf(COLOR_SECTION "Memory arena freed\n" COLOR_RESET);

    // End timing and print execution time
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf(COLOR_INFO "Total execution time: %.6f seconds\n" COLOR_RESET, elapsed_time);
    print_memory_usage();

    return 0;
}
