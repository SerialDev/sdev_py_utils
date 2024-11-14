// memory_arena.h

#ifndef MEMORY_ARENA_H
#define MEMORY_ARENA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * MemoryArena structure to manage a linear allocation of memory.
 */
typedef struct {
    uint8_t *base; ///< Base address of the arena.
    size_t size;   ///< Total size of the arena.
    size_t offset; ///< Current offset in the arena.
} MemoryArena;

/**
 * Initialize the memory arena.
 *
 * @param arena Pointer to the MemoryArena struct.
 * @param size  Total size of the memory arena in bytes.
 */
  //void MemoryArena_Init(MemoryArena *arena, size_t size);

/**
 * Allocate memory from the arena.
 *
 * @param arena Pointer to the MemoryArena struct.
 * @param size  Size of the allocation in bytes.
 * @return      Pointer to the allocated memory or NULL if out of memory.
 */
void *MemoryArena_Alloc(MemoryArena *arena, size_t size);

/**
 * Allocate aligned memory from the arena.
 *
 * @param arena Pointer to the MemoryArena struct.
 * @param size  Size of the allocation in bytes.
 * @param align Alignment in bytes (must be a power of two).
 * @return      Pointer to the allocated memory or NULL if out of memory.
 */
void *MemoryArena_AllocAligned(MemoryArena *arena, size_t size, size_t align);

/**
 * Reset the memory arena for reuse.
 *
 * @param arena Pointer to the MemoryArena struct.
 */
void MemoryArena_Reset(MemoryArena *arena);

/**
 * Free the memory arena and release resources.
 *
 * @param arena Pointer to the MemoryArena struct.
 */
void MemoryArena_Free(MemoryArena *arena);

/**
 * Duplicate a string into the arena.
 *
 * @param arena Pointer to the MemoryArena struct.
 * @param str   String to duplicate.
 * @return      Pointer to the duplicated string or NULL if out of memory.
 */
char *MemoryArena_StrDup(MemoryArena *arena, const char *str);

/**
 * Debug function to print the arena's current status.
 *
 * @param arena Pointer to the MemoryArena struct.
 */
void MemoryArena_Debug(MemoryArena *arena);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_ARENA_H

/* Implementation */

#ifdef MEMORY_ARENA_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void MemoryArena_Init(MemoryArena *arena, size_t size) {
    arena->base = (uint8_t *)malloc(size);
    if (!arena->base) {
        // Allocation failed
        fprintf(stderr, "MemoryArena_Init: Failed to allocate %zu bytes\n", size);
        arena->size = 0;
        arena->offset = 0;
        return;
    }
    arena->size = size;
    arena->offset = 0;
}

void *MemoryArena_Alloc(MemoryArena *arena, size_t size) {
    if (!arena->base) {
        fprintf(stderr, "MemoryArena_Alloc: Arena not initialized\n");
        return NULL;
    }
    if (arena->offset + size > arena->size) {
        // Out of memory
        fprintf(stderr, "MemoryArena_Alloc: Out of memory (requested %zu bytes)\n", size);
        return NULL;
    }
    void *ptr = arena->base + arena->offset;
    arena->offset += size;
    return ptr;
}

static size_t MemoryArena_AlignForward(size_t ptr, size_t align) {
    size_t modulo = ptr & (align - 1);
    if (modulo != 0) {
        ptr += align - modulo;
    }
    return ptr;
}



void *MemoryArena_AllocAligned(MemoryArena *arena, size_t size, size_t align) {
    if (!arena->base) {
        fprintf(stderr, "MemoryArena_AllocAligned: Arena not initialized\n");
        return NULL;
    }
    if ((align & (align - 1)) != 0) {
        fprintf(stderr, "MemoryArena_AllocAligned: Alignment must be a power of two\n");
        return NULL;
    }

    size_t current_ptr = (size_t)(arena->base + arena->offset);
    size_t offset = MemoryArena_AlignForward(current_ptr, align) - (size_t)arena->base;

    if (offset + size > arena->size) {
        // Out of memory
        fprintf(stderr, "MemoryArena_AllocAligned: Out of memory (requested %zu bytes with alignment %zu)\n", size, align);
        return NULL;
    }
    void *ptr = arena->base + offset;
    arena->offset = offset + size;
    return ptr;
}

void MemoryArena_Reset(MemoryArena *arena) {
    arena->offset = 0;
}

void MemoryArena_Free(MemoryArena *arena) {
    if (arena->base) {
        free(arena->base);
        arena->base = NULL;
        arena->size = 0;
        arena->offset = 0;
    }
}

char *MemoryArena_StrDup(MemoryArena *arena, const char *str) {
    size_t len = strlen(str) + 1; // Include null terminator
    char *copy = (char *)MemoryArena_Alloc(arena, len);
    if (copy) {
        memcpy(copy, str, len);
    }
    return copy;
}

void MemoryArena_Debug(MemoryArena *arena) {
    printf("MemoryArena Debug - Total Size: %zu bytes, Used: %zu bytes, Free: %zu bytes\n",
           arena->size, arena->offset, arena->size - arena->offset);
}

#endif // MEMORY_ARENA_IMPLEMENTATION
