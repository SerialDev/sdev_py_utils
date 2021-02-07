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



# Install 

local development
```
python install_development

```

local production
```
python install_production
```

remote
```
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_algo_utils\&subdirectory=sdev_algo_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_concurrency_utils\&subdirectory=sdev_concurrency_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_decorator_utils\&subdirectory=sdev_decorator_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_introspection_utils\&subdirectory=sdev_introspection_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_lib_utils\&subdirectory=sdev_lib_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_logging_utils\&subdirectory=sdev_logging_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_misc_utils\&subdirectory=sdev_misc_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_pycore_utils\&subdirectory=sdev_pycore_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_pytech_utils\&subdirectory=sdev_pytech_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_scipy_utils\&subdirectory=sdev_scipy_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_search_utils\&subdirectory=sdev_search_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_viz_utils\&subdirectory=sdev_viz_utils
pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_network_analysis\&subdirectory=sdev_network_analysis
```

```
pip uninstall sdev_algo_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_algo_utils\&subdirectory=sdev_algo_utils
pip uninstall sdev_concurrency_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_concurrency_utils\&subdirectory=sdev_concurrency_utils
pip uninstall sdev_decorator_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_decorator_utils\&subdirectory=sdev_decorator_utils
pip uninstall sdev_introspection_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_introspection_utils\&subdirectory=sdev_introspection_utils
pip uninstall sdev_lib_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_lib_utils\&subdirectory=sdev_lib_utils
pip uninstall sdev_logging_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_logging_utils\&subdirectory=sdev_logging_utils
pip uninstall sdev_misc_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_misc_utils\&subdirectory=sdev_misc_utils
pip uninstall sdev_pycore_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_pycore_utils\&subdirectory=sdev_pycore_utils
pip uninstall sdev_pytech_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_pytech_utils\&subdirectory=sdev_pytech_utils
pip uninstall sdev_scipy_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_scipy_utils\&subdirectory=sdev_scipy_utils
pip uninstall sdev_search_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_search_utils\&subdirectory=sdev_search_utils
pip uninstall sdev_viz_utils -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_viz_utils\&subdirectory=sdev_viz_utils
pip uninstall sdev_network_analysis -y && pip install git+https://github.com/SerialDev/sdev_py_utils.git#egg=sdev_network_analysis\&subdirectory=sdev_network_analysis
```
