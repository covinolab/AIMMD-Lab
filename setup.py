'''
setup.py for AIMMD
this file was created with the help of ChatGPT
'''

from setuptools import setup, find_packages

# read README for PyPI long description if present
def readme():
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return 'AIMMD - AI for molecular mechanism discovery.'

setup(
    name='aimmd',
    version='0.1.0',
    description=(
        'AI for molecular mechanism discovery for data generation and analysis'
        'of molecular systems characterized by rare-event transitions.'),
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/gl95/aimmd',  # replace with your repo URL
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy',
        'MDAnalysis',
        'mdtraj',
        'torch',
        'matplotlib',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    keywords=('molecular-dynamics ai enhanced-sampling gromacs '
              'biophysics rare-event-transitions'),
    include_package_data=True,
    zip_safe=False,
)
