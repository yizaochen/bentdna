from setuptools import setup, find_packages

setup(name='bentdna', 
      version='0.1',
      packages=find_packages(),
      url='https://github.com/yizaochen/bentdna.git',
      author='Yizao Chen',
      author_email='yizaochen@gmail.com',
      license='MIT',
      install_requires=[
          'pandas',
          'numpy',
          'MDAnalysis',
          'plotly'
      ]
      )