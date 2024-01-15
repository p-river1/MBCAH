from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

packages = [
    'numpy',
    'scipy',
    'pandas',
    'scikit-learn',
    'joblib',
    'matplotlib',
    'seaborn',
    'notebook',
    'tqdm',
]

setup(
    name='mbcah',
    version='0.1.0',
    description=(
        'Model-based clustering and alignment of water quality'
        ' curves with prior knowledge integration using'
        ' hidden Markov random fields'
    ),
    author='Paul Riverain',
    author_email='paul.riverain@univ-eiffel.com',
    install_requires=packages,
    long_description=long_description,
    long_description_content_type='text/markdown',
    tests_require=None,
    packages=find_packages()
)
