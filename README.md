# sdev_py_utils


sphinx-apidoc -o .\docs\build\  .\
sphinx-apidoc -o .\docs\build\  .\modular_templating
sphinx-apidoc -o .\docs\build\  .\sdev_algo_utils
sphinx-apidoc -o .\docs\build\  .\sdev_concurrency_utils
sphinx-apidoc -o .\docs\build\  .\sdev_decorator_utils
sphinx-apidoc -o .\docs\build\  .\sdev_introspection_utils
sphinx-apidoc -o .\docs\build\  .\sdev_lib_utils
sphinx-apidoc -o .\docs\build\  .\sdev_logging_utils
sphinx-apidoc -o .\docs\build\  .\sdev_misc_utils
sphinx-apidoc -o .\docs\build\  .\sdev_network_analysis
sphinx-apidoc -o .\docs\build\  .\sdev_pycore_utils
sphinx-apidoc -o .\docs\build\  .\sdev_pytech_utils
sphinx-apidoc -o .\docs\build\  .\sdev_scipy_utils
sphinx-apidoc -o .\docs\build\  .\sdev_search_utils
sphinx-apidoc -o .\docs\build\  .\sdev_viz_utils

sphinx-build -b html .\docs .\docs\build\
