import setuptools # necessary for install_requires

from distutils.core import Command
from numpy.distutils.core import Extension
from numpy.distutils.command.build_clib import build_clib
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.command.sdist import sdist
from numpy.distutils.command.build import build
from distutils.command.clean import clean
from Cython.Build import cythonize

import sys
import os
import numpy as np
import shutil

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))


def find_version(path, name='version'):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^%s = ['\"]([^'\"]*)['\"]" %name,
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('version not found')


CLASS_VERSION = find_version('pyclass/version.py', name='class_version')


def build_CLASS(prefix):
    """
    Function to dowwnload CLASS from github and build the library
    """
    # latest class version and download link
    args = (package_basedir, package_basedir, CLASS_VERSION, os.path.abspath(prefix))
    command = 'sh %s/depends/install_class.sh %s %s %s' %args

    ret = os.system(command)
    if ret != 0:
        raise ValueError('could not build CLASS v%s' %CLASS_VERSION)


class build_external_clib(build_clib):
    """
    Custom command to build CLASS first, and then GCL library
    """
    def finalize_options(self):
        build_clib.finalize_options(self)

        # create the CLASS build directory and save the include path
        self.class_build_dir = self.build_temp
        self.include_dirs.insert(0, os.path.join(self.class_build_dir, 'include'))
        for external in ['heating','HyRec2020','RecfastCLASS']:
            self.include_dirs.insert(0, os.path.join(self.class_build_dir, 'external', external))

    def build_libraries(self, libraries):
        # build CLASS first
        build_CLASS(self.class_build_dir)

        # update the link objects with CLASS library
        link_objects = ['libclass.a']
        #link_objects = list(glob(os.path.join(self.class_build_dir, '*', 'libclass.a')))

        self.compiler.set_link_objects(link_objects)
        self.compiler.library_dirs.insert(0, os.path.join(self.class_build_dir, 'lib'))

        # then no longer need to build class.

        libraries = [lib for lib in libraries if lib[0] != 'class']

        for (library, build_info) in libraries:

            # check swig version
            if library == 'gcl' and swig_needed:
                check_swig_version()

            # update include dirs
            self.include_dirs += build_info.get('include_dirs', [])

        build_clib.build_libraries(self, libraries)


class custom_build_ext(build_ext):
    """
    Custom extension building to grab include directories
    from the ``build_clib`` command
    """
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.include_dirs.append(np.get_include())
        self.cython_directives = {'language_level': '3' if sys.version_info.major>=3 else '2'}

    def run(self):
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            self.include_dirs += build_clib.include_dirs
            self.library_dirs += build_clib.compiler.library_dirs

        # copy data files from temp to pyclass package directory
        for name in ['external','data']:
            shutil.rmtree(os.path.join(self.build_lib, 'pyclass', name), ignore_errors=True)
            shutil.copytree(os.path.join(self.build_temp, name), os.path.join(self.build_lib, 'pyclass', name))

        build_ext.run(self)


class custom_sdist(sdist):

    def run(self):
        from six.moves.urllib import request

        # download CLASS
        tarball_link = 'https://github.com/lesgourg/class_public/archive/v%s.tar.gz' %CLASS_VERSION
        tarball_local = os.path.join('depends', 'class-v%s.tar.gz' %CLASS_VERSION)
        request.urlretrieve(tarball_link, tarball_local)

        # run the default
        sdist.run(self)


class custom_clean(clean):

    def run(self):

        # run the built-in clean
        clean.run(self)

        # remove the CLASS tmp directories
        os.system('rm -rf depends/tmp*')

        # remove build directory
        if os.path.exists('build'):
            shutil.rmtree('build')


def libclass_config():
    return ('class', {})


def classy_extension_config():

    # the configuration for GCL python extension
    config = {}
    config['name'] = 'pyclass.binding'
    config['extra_link_args'] = ['-g', '-fPIC','-lgomp']
    config['extra_compile_args'] = []
    # important or get a symbol not found error, because class is
    # compiled with c++?
    config['language'] = 'c'
    config['libraries'] = ['class', 'gfortran', 'm']

    # determine if swig needs to be called
    config['sources'] = ['pyclass/binding.pyx']

    return config


if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(name='pyclass',
          version=find_version('pyclass/version.py'),
          author='Arnaud de Mattia, based on classylss of Nick Hand, Yu Feng',
          author_email='',
          description='Python binding of the CLASS CMB Boltzmann code',
          license='GPL3',
          url='http://github.com/adematti/pyclass',
          install_requires=['numpy', 'cython'],
          ext_modules = cythonize([
                        Extension(**classy_extension_config())
          ]),
          libraries=[libclass_config()],
          cmdclass = {
              'sdist': custom_sdist,
              'build_clib': build_external_clib,
              'build_ext': custom_build_ext,
              'clean': custom_clean
          },
         packages=['pyclass']
    )
