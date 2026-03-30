# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Modeling Agent'
copyright = '2026, BESSER-PEARL'
author = 'BESSER-PEARL'

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

# -- Logo and branding
html_logo = '_static/besser_logo.png'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'style_nav_header_background': '#1a1a2e',
}
html_favicon = '_static/besser_ico.ico'

html_context = {
    'display_github': True,
    'github_user': 'BESSER-PEARL',
    'github_repo': 'modeling-agent',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
