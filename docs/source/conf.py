# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Modeling Agent'
copyright = '2023, Luxembourg Institute of Science and Technology (LIST)'
author = 'list-of-authors'

release = '0.3'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]

# -- Mermaid configuration
mermaid_d3_zoom = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_context = {
    'display_github': True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
