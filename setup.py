from setuptools import setup
import os,glob,warnings,sys,fnmatch
from setuptools.command.test import test as TestCommand
from distutils.core import setup
import numpy.distutils.misc_util


if sys.version_info < (3,0):
    sys.exit('Sorry, Python 2 is not supported')

# class SNTDTest(TestCommand):

#     def run_tests(self):
#         import sntd
#         errno = sntd.test()
#         sntd.test_sntd()
#         sys.exit(errno)

AUTHOR = 'Justin Pierel'
AUTHOR_EMAIL = 'jr23@email.sc.edu'
VERSION = '0.0.1'
LICENSE = 'BSD'
URL = ''

def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

PACKAGENAME='pierel_util'




setup(
    name=PACKAGENAME,
    #cmdclass={'test': SNTDTest},
    setup_requires=['numpy'],
    install_requires=['matplotlib'],
    packages=[PACKAGENAME],
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    #package_data={'sntd':data_files},
    #include_package_data=True
)
