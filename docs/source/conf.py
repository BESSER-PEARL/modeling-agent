# Configuration file for the Sphinx documentation builder.
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -- Project information

project = 'Modeling Agent'
copyright = '2026, BESSER-PEARL'
author = 'BESSER-PEARL'

release = '8.0.0'
version = '8.0'
html_title = 'BESSER Modeling Agent'

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

html_theme = "sphinx_immaterial"
extensions.append("sphinx_immaterial")
html_logo = "_static/besser_logo_dark.png"
html_static_path = ["_static"]
html_css_files = ["docs.css"]
html_theme_options = {
    "font": False,
    "features": ["navigation.sections", "navigation.top", "search.highlight"],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "cyan",
            "accent": "cyan",
            "toggle": {"icon": "material/weather-night", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "cyan",
            "accent": "cyan",
            "toggle": {"icon": "material/weather-sunny", "name": "Switch to light mode"},
        },
    ],
    "repo_url": "https://github.com/BESSER-PEARL/modeling-agent",
    "repo_name": "Source",
    "globaltoc_collapse": True,
    "toc_title": "On this page",
}
html_favicon = "_static/besser_ico.ico"
html_show_sourcelink = False
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_context = {
    'display_github': True,
    'github_user': 'BESSER-PEARL',
    'github_repo': 'modeling-agent',
    'github_version': 'develop',
    'conf_py_path': '/docs/source/',
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
