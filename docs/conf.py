import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'depyf'
author = 'Kaichao You et al.'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'sphinx.ext.todo',
]


source_encoding = 'utf-8'

master_doc = 'index'
html_theme = "sphinx_rtd_theme"
language = "en-US"

todo_include_todos = True

html_static_path = ["_static"]

html_logo = "_static/images/depyf-logo.svg"

html_css_files = [
    'css/style.css',
]

highlight_language = 'python3'

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 1,  # Adjust as needed
    'includehidden': True,
    'titles_only': False
}
