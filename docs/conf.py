project = 'sdev_utils'
copyright = '2019, Andres Mariscal'
author = 'Andres Mariscal[serialdev]'


version = ''
# The full version, including alpha/beta/rc tags
release = ''

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

extensions = [
'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]
source_suffix = '.rst'
master_doc = 'index'
language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

html_theme = 'nature'
html_static_path = ['_static']

htmlhelp_basename = 'Serialdev Utilities docs'

