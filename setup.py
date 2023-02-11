from setuptools import setup

setup(name='kcmc',
      version='0.1',
      description='Kernel Conditional Moment Constraints for Confounding Robust Inference',
      url='http://github.com/kstoneriv3/kcmc',
      author='Kei Ishikawa',
      author_email='k.stoneriv@gmail.com',
      license='MIT',
      packages=['kcmc'],
      install_requires=[
          "cvxpy", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "torch", "tqdm"
      ],
      zip_safe=False)
