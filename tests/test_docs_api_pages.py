"""The API reference covers every public module of the C-GAIMMD and analysis packages.

A module is public when no part of its dotted name starts with an underscore.
Each one must be named by an ``.. automodule::`` directive on a page under
``docs/source/api``, so that a new module cannot be merged without a place in the
reference (the docs build itself does not notice a module that no page names).
The modules are found from the files; nothing is imported.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_PAGES = ROOT / 'docs' / 'source' / 'api'
PACKAGES = ('aimmd/cgaimmd', 'aimmd/analysis')


def _documented_modules():
    names = set()
    for page in API_PAGES.glob('*.rst'):
        names |= set(re.findall(r'^\.\. automodule:: +(\S+)', page.read_text(), re.MULTILINE))
    return names


def _public_modules(package):
    """Dotted names of the public modules and subpackages below `package` (not `package` itself)."""
    names = []
    for path in sorted((ROOT / package).rglob('*.py')):
        parts = path.relative_to(ROOT).with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        if any(part.startswith('_') for part in parts):
            continue
        if parts != tuple(package.split('/')):
            names.append('.'.join(parts))
    return names


@pytest.mark.parametrize('package', PACKAGES)
def test_every_public_module_has_an_api_page(package):
    modules = _public_modules(package)
    assert modules, f'no modules found below {package}'
    missing = sorted(set(modules) - _documented_modules())
    assert not missing, (f'not named by any ".. automodule::" under docs/source/api: {missing}; '
                         f'add them to the page of their package')
