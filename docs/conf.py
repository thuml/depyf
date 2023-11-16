import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'depyf'
author = 'Kaichao You et al.'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

source_encoding = 'utf-8'

master_doc = 'index'
language = "en-US"

todo_include_todos = True
