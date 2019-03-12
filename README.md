# sdev_py_utils


sphinx-apidoc -o .\docs  .\
sphinx-apidoc -o .\docs  .\modular_templating
sphinx-apidoc -o .\docs  .\sdev_algo_utils
sphinx-apidoc -o .\docs  .\sdev_concurrency_utils
sphinx-apidoc -o .\docs  .\sdev_decorator_utils
sphinx-apidoc -o .\docs  .\sdev_introspection_utils
sphinx-apidoc -o .\docs  .\sdev_lib_utils
sphinx-apidoc -o .\docs  .\sdev_logging_utils
sphinx-apidoc -o .\docs  .\sdev_misc_utils
sphinx-apidoc -o .\docs  .\sdev_network_analysis
sphinx-apidoc -o .\docs  .\sdev_pycore_utils
sphinx-apidoc -o .\docs  .\sdev_pytech_utils
sphinx-apidoc -o .\docs  .\sdev_scipy_utils
sphinx-apidoc -o .\docs  .\sdev_search_utils
sphinx-apidoc -o .\docs  .\sdev_viz_utils

sphinx-build -b html .\docs .\docs\build\
