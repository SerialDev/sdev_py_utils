import os

def generate_folder_structure(module_name):
    """
    Generate new module folder structure

    Parameters
    ----------

    module_name : string
       Name of the module to create
    Returns
    -------

    int
        Returns 1 on success
    Raises
    ------

    FileExistsError
        current file already exists
    """
    os.mkdir(os.path.join(os.getcwd(), module_name))
    os.mkdir(os.path.join(os.getcwd(), module_name, module_name))
    return 1

def generate_setup(module_name, author, author_email, description,
                   version="0.1", url="", license='MIT'):

    with open(os.path.join(os.getcwd(), module_name,"setup.py"), 'w') as f:
        f.write(f"""
from setuptools import setup

setup(name = '{module_name}',
      version ='{version}',
      description='{description}',
      url='{url}',
      author='{author}',
      author_email='{author_email}',
      license='{license}',
      packages=['{module_name}'],
      zip_safe=False)
        """)
    return 1

def generate_importables(module_name):
    with open(os.path.join(os.getcwd(), module_name, module_name, "__init__.py"), 'w') as f:
        f.write("from .example_module import *")

    with open(os.path.join(os.getcwd(), module_name, module_name, "example_module.py"), 'w') as f:
        f.write('print("Success")')
