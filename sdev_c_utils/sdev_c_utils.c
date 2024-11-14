
#include <Python.h>
#include <stdlib.h>
#define MEMORY_ARENA_IMPLEMENTATION
#include "memory_arena.h"

static int compare(const void *a, const void *b) {
    double da = *((const double *)a), db = *((const double *)b);
    return (da > db) - (da < db);
}

static PyObject *method_median_arena(PyObject *self, PyObject *args) {
    PyObject *listObj;
    MemoryArena arena;
    MemoryArena_Init(&arena, 1024 * 1024);

    if (!PyArg_ParseTuple(args, "O", &listObj) || !PyList_Check(listObj)) {
        MemoryArena_Free(&arena);
        return NULL;
    }

    Py_ssize_t num = PyList_Size(listObj);
    if (num == 0) {
        MemoryArena_Free(&arena);
        return NULL;
    }

    size_t required_size = num * sizeof(double);
    if (required_size > arena.size) {
        MemoryArena_Free(&arena);
        MemoryArena_Init(&arena, required_size);
    }

    double *values = (double *)MemoryArena_Alloc(&arena, num * sizeof(double));
    if (!values) {
        MemoryArena_Free(&arena);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < num; i++) {
        PyObject *temp = PyList_GetItem(listObj, i);
        if (!PyNumber_Check(temp)) {
            MemoryArena_Free(&arena);
            return NULL;
        }
        values[i] = PyFloat_AsDouble(temp);
    }

    qsort(values, num, sizeof(double), compare);

    double median;
    if (num % 2 == 0)
        median = (values[num / 2 - 1] + values[num / 2]) / 2.0;
    else
        median = values[num / 2];

    MemoryArena_Free(&arena);
    return Py_BuildValue("d", median);
}

static PyObject *method_add(PyObject *self, PyObject *args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    return Py_BuildValue("i", a + b);
}

static PyObject *create_arena(PyObject *self, PyObject *args) {
    size_t size;
    if (!PyArg_ParseTuple(args, "n", &size)) {
        return NULL;
    }
    MemoryArena *arena = (MemoryArena *)malloc(sizeof(MemoryArena));
    MemoryArena_Init(arena, size);
    return PyCapsule_New((void *)arena, "MemoryArena", NULL);
}

static PyObject *free_arena(PyObject *self, PyObject *args) {
    PyObject *arenaCapsule;
    if (!PyArg_ParseTuple(args, "O", &arenaCapsule)) {
        return NULL;
    }
    MemoryArena *arena = (MemoryArena *)PyCapsule_GetPointer(arenaCapsule, "MemoryArena");
    if (!arena) {
        return PyErr_Format(PyExc_RuntimeError, "Invalid memory arena");
    }
    MemoryArena_Free(arena);
    free(arena);
    Py_RETURN_NONE;
}

static PyObject *uniquify_to_dict(PyObject *self, PyObject *args) {
    // 2x speedup
  
    PyObject *iterable;
    if (!PyArg_ParseTuple(args, "O", &iterable)) {
        return NULL;
    }

    PyObject *iterator = PyObject_GetIter(iterable);
    if (!iterator) {
        PyErr_SetString(PyExc_TypeError, "The argument must be an iterable");
        return NULL;
    }

    PyObject *result_dict = PyDict_New();
    if (!result_dict) {
        Py_DECREF(iterator);
        return NULL;
    }

    PyObject *current_key = NULL;
    PyObject *current_list = NULL;

    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        if (!PyTuple_Check(item) || PyTuple_Size(item) != 2) {
            Py_DECREF(item);
            Py_DECREF(iterator);
            Py_DECREF(result_dict);
            PyErr_SetString(PyExc_TypeError, "Each item in the iterable must be a tuple of size 2");
            return NULL;
        }

        PyObject *x = PyTuple_GetItem(item, 0);  // Borrowed reference
        PyObject *y = PyTuple_GetItem(item, 1);  // Borrowed reference

        if (current_key && PyObject_RichCompareBool(x, current_key, Py_EQ)) {
            // x is the same as the current_key, add y to current_list
            PyList_Append(current_list, y);
        } else {
            // New key encountered, add current list to result_dict
            if (current_key) {
                PyDict_SetItem(result_dict, current_key, current_list);
                Py_DECREF(current_list);
                Py_DECREF(current_key);
            }
            // Initialize a new list for the new key
            current_list = PyList_New(0);
            if (!current_list) {
                Py_DECREF(item);
                Py_DECREF(iterator);
                Py_DECREF(result_dict);
                return NULL;
            }
            PyList_Append(current_list, y);
            Py_INCREF(x);  // Increment ref count as we're storing x
            current_key = x;
        }
        Py_DECREF(item);
    }
    // Handle the last key-value pair
    if (current_key) {
        PyDict_SetItem(result_dict, current_key, current_list);
        Py_DECREF(current_list);
        Py_DECREF(current_key);
    }

    Py_DECREF(iterator);
    return result_dict;
}


static PyMethodDef SdevCUtilsMethods[] = {
    {"add", method_add, METH_VARARGS, "Add two numbers"},
    {"create_arena", create_arena, METH_VARARGS, "Create a new memory arena"},
    {"free_arena", free_arena, METH_VARARGS, "Free the memory arena stored in a PyObject"},
    {"median_arena", method_median_arena, METH_VARARGS, "Calculate the median of a list of numbers"},
    {"uniquify_to_dict", uniquify_to_dict, METH_VARARGS, "Uniquify values in an iterator into a dictionary."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sdev_c_utilsmodule = {
    PyModuleDef_HEAD_INIT,
    "sdev_c_utils",
    NULL,
    -1,
    SdevCUtilsMethods
};

PyMODINIT_FUNC PyInit_sdev_c_utils(void) {
    return PyModule_Create(&sdev_c_utilsmodule);
}
