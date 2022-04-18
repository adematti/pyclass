import os
import sys
import glob
import shutil
import sysconfig
import setuptools  # necessary for install_requires

from numpy.distutils.core import Extension
from numpy.distutils.command.build_clib import build_clib
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.command.sdist import sdist
from numpy.distutils.command.develop import develop
from distutils.command.clean import clean
from Cython.Build import cythonize
import numpy as np

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyclass'


sys.path.insert(0, os.path.join(package_basedir, package_basename))
import _version
version = _version.version
class_version = _version.class_version


def build_CLASS(prefix):
    """
    Function to dowwnload CLASS from github and build the library
    """
    # latest class version and download link
    args = (package_basedir, package_basedir, class_version, os.path.abspath(prefix))
    command = 'sh {}/depends/install_class.sh {} {} {}'.format(*args)

    ret = os.system(command)
    if ret != 0:
        raise ValueError('could not build CLASS v{}'.format(class_version))


class custom_build_clib(build_clib):
    """
    Custom command to build CLASS first, and then GCL library
    """
    def finalize_options(self):
        build_clib.finalize_options(self)

        # create the CLASS build directory and save the include path
        self.class_build_dir = self.build_temp
        self.include_dirs.insert(0, os.path.join(self.class_build_dir, 'include'))
        for external in ['heating', 'HyRec2020', 'RecfastCLASS']:
            self.include_dirs.insert(0, os.path.join(self.class_build_dir, 'external', external))

    def build_libraries(self, libraries):
        # build CLASS first
        build_CLASS(self.class_build_dir)

        # update the link objects with CLASS library
        link_objects = ['libclass.a']
        # link_objects = list(glob(os.path.join(self.class_build_dir, '*', 'libclass.a')))

        self.compiler.set_link_objects(link_objects)
        self.compiler.library_dirs.insert(0, os.path.join(self.class_build_dir, 'lib'))

        # then no longer need to build class.

        libraries = [lib for lib in libraries if lib[0] != 'class']

        for (library, build_info) in libraries:
            # update include dirs
            self.include_dirs += build_info.get('include_dirs', [])

        super(custom_build_clib, self).build_libraries(libraries)


class custom_build_ext(build_ext):
    """Custom extension building to grab include directories from the ``build_clib`` command."""

    def finalize_options(self):
        build_ext.finalize_options(self)
        self.include_dirs.append(np.get_include())
        self.cython_directives = {'language_level': '3' if sys.version_info.major >= 3 else '2'}

    def run(self):
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            self.include_dirs += build_clib.include_dirs
            self.library_dirs += build_clib.compiler.library_dirs

        # copy data files from temp to pyclass package directory
        for name in ['external', 'data']:
            shutil.rmtree(os.path.join(self.build_lib, 'pyclass', name), ignore_errors=True)
            shutil.copytree(os.path.join(self.build_temp, name), os.path.join(self.build_lib, 'pyclass', name))

        super(custom_build_ext, self).run()


class custom_sdist(sdist):

    def run(self):
        from six.moves.urllib import request

        # download CLASS
        tarball_link = 'https://github.com/adematti/class_public/archive/v{}.tar.gz'.format(class_version)
        tarball_local = os.path.join('depends', 'class-v{}.tar.gz'.format(class_version))
        request.urlretrieve(tarball_link, tarball_local)

        # run the default
        super(custom_sdist, self).run()


class custom_develop(develop):

    def run(self):
        self.run_command('build_ext')
        build_ext = self.get_finalized_command('build_ext')
        for name in ['external', 'data']:
            shutil.copytree(os.path.join(build_ext.build_temp, name), os.path.join(package_basedir, 'pyclass', name))
        super(custom_develop, self).run()


class custom_clean(clean):

    def run(self):

        # run the built-in clean
        super(custom_clean, self).run()

        # remove the CLASS tmp directories
        for dirpath in glob.glob(os.path.join('depends', 'tmp*')):
            if os.path.exists(dirpath) and os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
        # remove external and data directories set by develop
        for name in ['external', 'data']:
            shutil.rmtree(os.path.join(package_basedir, 'pyclass', name), ignore_errors=True)
        for fn in glob.glob(os.path.join(package_basedir, 'pyclass', 'binding.c*')):
            try: os.remove(fn)
            except OSError: pass
        # remove build directory
        shutil.rmtree('build', ignore_errors=True)


def libclass_config():
    return ('class', {})


def find_compiler():
    compiler = os.getenv('CC', None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC', None)
    return compiler


def classy_extension_config():

    compiler = find_compiler()
    # the configuration for GCL python extension
    config = {}
    config['name'] = 'pyclass.binding'
    config['extra_link_args'] = ['-g', '-fPIC']
    config['extra_compile_args'] = []
    # important or get a symbol not found error, because class is
    # compiled with c++?
    config['language'] = 'c'
    config['libraries'] = ['class', 'm']

    # determine if swig needs to be called
    config['sources'] = [os.path.join('pyclass', 'binding.pyx')]

    os.environ.setdefault('CC', compiler)
    if compiler == 'clang':
        # see https://github.com/lesgourg/class_public/issues/405
        os.environ.setdefault('OMPFLAG', '-Xclang -fopenmp')
        config['extra_link_args'] += ['-lomp']
    else:
        config['extra_link_args'] += ['-lgomp']
    if compiler in ['cc', 'icc']:
        # see https://github.com/lesgourg/class_public/issues/40
        config['libraries'] += ['irc', 'svml', 'imf']
        config['extra_link_args'] += ['-liomp5']

    return config


if __name__ == '__main__':

    from numpy.distutils.core import setup

    setup(name=package_basename,
          version=version,
          author='Arnaud de Mattia, based on classylss by Nick Hand, Yu Feng',
          author_email='',
          description='Python binding of the CLASS CMB Boltzmann code',
          license='GPL3',
          url='http://github.com/adematti/pyclass',
          install_requires=['numpy', 'cython'],
          ext_modules=cythonize([Extension(**classy_extension_config())]),
          libraries=[libclass_config()],
          cmdclass={'sdist': custom_sdist,
                    'build_clib': custom_build_clib,
                    'build_ext': custom_build_ext,
                    'develop': custom_develop,
                    'clean': custom_clean
                    },
          packages=[package_basename])
