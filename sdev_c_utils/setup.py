from distutils.core import setup, Extension

module = Extension("sdev_c_utils", sources=["sdev_c_utils.c"])

setup(
    name="sdev_utils",
    version="0.1",
    description="C-gen-modules",
    ext_modules=[module],
    author="Andres",
    author_email="carlos.mariscal.melgar@gmail.com",
    license="GPL3",
    zip_safe=False,
)
