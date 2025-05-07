import os
import sys
import glob
import shutil
import sysconfig

from setuptools import setup, Extension, find_packages
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from setuptools.command.develop import develop
from distutils.command.clean import clean


# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyclass'


sys.path.insert(0, os.path.join(package_basedir, package_basename))
import _version
version = _version.version


def find_branches():
    branches = find_packages(where=package_basename)
    select = os.getenv('PYCLASS_BRANCHES', None)
    if select is not None:
        branches = [branch for branch in branches if branch in select]
    return branches


def load_version(branch):
    import importlib
    spec = importlib.util.spec_from_file_location(branch, os.path.join(package_basedir, package_basename, branch, '_version.py'))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


def find_url(branch):
    foo = load_version(branch)
    return foo.url


def find_include(branch):
    foo = load_version(branch)
    return list(getattr(foo, 'include', ['include']))


def download(url, target, authorization=None, size=None):
    """
    Download file from input ``url``.

    Parameters
    ----------
    url : str, Path
        url to download file from.

    target : str, Path
        Path where to save the file, on disk.

    size : int, default=None
        Expected file size, in bytes, used to show progression bar.
        If not provided, taken from header (if the file is larger than a couple of GBs,
        it may be wrong due to integer overflow).
        If a sensible file size is obtained, a progression bar is printed.
    """
    # Adapted from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    print('Downloading {} to {}.'.format(url, target))
    import requests
    # See https://stackoverflow.com/questions/61991164/python-requests-missing-content-length-response
    headers = {}
    if authorization:
        headers.update({'Authorization': authorization})
    if size is None:
        size = requests.head(url, headers={**headers, 'Accept-Encoding': None}).headers.get('content-length')
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, stream=True)
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return False

    with open(target, 'wb') as file:
        if size is None or int(size) < 0:  # no content length header
            file.write(r.content)
        else:
            import shutil
            width = shutil.get_terminal_size((80, 20))[0] - 9  # pass fallback
            dl, size, current = 0, int(size), 0
            for data in r.iter_content(chunk_size=2048):
                dl += len(data)
                file.write(data)
                if size:
                    frac = min(dl / size, 1.)
                    done = int(width * frac)
                    if done > current:  # it seems, when content-length is not set iter_content does not care about chunk_size
                        print('\r[{}{}] [{:3.0%}]'.format('#' * done, ' ' * (width - done), frac), end='', flush=True)
                        current = done
            print('')
    return True


def build_class(build_dir, branch='base'):
    """Function to dowwnload CLASS from github and build the library."""
    # latest class version and download link
    url = find_url(branch)
    authorization = os.getenv('AUTHORIZATION', None)
    depends_dir = os.path.join(package_basedir, 'depends')
    tarball = 'tmp-class-{}.tar.gz'.format(branch)
    if not download(url, target=os.path.join(depends_dir, tarball), authorization=authorization):
        print('\033[93mCould not access {}; skipping branch {}.\033[0m'.format(url, branch))
        #import warnings
        #warnings.warn('Could not access {}; skipping branch {}.'.format(url, branch))
        return False
    patch = os.path.join(os.path.join(package_basedir, package_basename, branch, 'patch'))
    args = (depends_dir, tarball, patch, os.path.abspath(build_dir), ' '.join(find_include(branch)))
    command = 'cd {}; TARBALL="{}" PATCH={} DEST={} INCLUDE="{}" make install'.format(*args)
    if os.system(command) != 0:
        raise ValueError('could not build CLASS {}'.format(build_dir))
    return True


class custom_build_ext(build_ext):
    """Custom extension building to grab include directories from the ``build_ext`` command."""

    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy as np
        self.include_dirs.append(np.get_include())
        self.cython_directives = {'language_level': '3' if sys.version_info.major >= 3 else '2'}

    def run(self):
        nobuild = []
        for extension in self.extensions:
            branch = extension.name[len(package_basename) + 1:-len('binding') - 1]
            build_dir = os.path.join(self.build_temp, branch)
            if build_class(build_dir, branch=branch):
                library_dir = os.path.join(build_dir, 'lib')
                # os.rename(os.path.join(library_dir, 'libclass.a'), os.path.join(library_dir, 'libclass-{}.a'.format(branch)))
                for include in find_include(branch):
                    extension.include_dirs.insert(0, os.path.join(build_dir, include))

                for external in ['heating', 'HyRec2020', 'RecfastCLASS']:
                    extension.include_dirs.insert(0, os.path.join(build_dir, 'external', external))
                extension.library_dirs.insert(0, library_dir)
                extension.libraries.insert(0, 'class')
                # extension.include_dirs = self.include_dirs + extension.include_dirs

                # copy data files from temp to pyclass package directory
                for name in ['external', 'data']:
                    shutil.rmtree(os.path.join(self.build_lib, package_basename, branch, name), ignore_errors=True)
                    shutil.copytree(os.path.join(self.build_temp, branch, name), os.path.join(self.build_lib, package_basename, branch, name))
            else:
                nobuild.append(extension)
        for extension in nobuild:
            del self.extensions[self.extensions.index(extension)]

        # self.include_dirs.clear()
        super(custom_build_ext, self).run()


class custom_develop(develop):

    def run(self):
        build_ext = self.get_finalized_command('build_ext')
        super(custom_develop, self).run()
        for branch in find_branches():
            for name in ['external', 'data']:
                shutil.rmtree(os.path.join(package_basedir, package_basename, branch, name), ignore_errors=True)
                try: shutil.copytree(os.path.join(build_ext.build_temp, branch, name), os.path.join(package_basedir, package_basename, branch, name))
                except OSError: pass


class custom_clean(clean):

    def run(self):

        # run the built-in clean
        super(custom_clean, self).run()

        # remove CLASS tmp directories
        for dirpath in glob.glob(os.path.join(package_basedir, 'depends', 'tmp*')):
            if os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
            else:
                os.remove(dirpath)
        # remove external and data directories set by develop
        for branch in find_branches():
            for name in ['external', 'data']:
                shutil.rmtree(os.path.join(package_basedir, package_basename, branch, name), ignore_errors=True)
            for fn in glob.glob(os.path.join(package_basedir, package_basename, branch, 'binding.c*')):
                try: os.remove(fn)
                except OSError: pass
        # remove build directory
        shutil.rmtree('build', ignore_errors=True)
        shutil.rmtree('dist', ignore_errors=True)
        shutil.rmtree(package_basename + '.egg-info', ignore_errors=True)


def find_compiler():
    compiler = os.getenv('CC', None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC', None)
    import platform
    uname = platform.uname().system
    if compiler is None:
        compiler = 'gcc'
        if uname == 'Darwin': compiler = 'clang'
    return compiler


def compiler_is_clang(compiler):
    if compiler == 'clang':
        return True
    import subprocess
    proc = subprocess.Popen([compiler, '--version'], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    if 'clang' in out:
        return True
    return False


def classy_extension_config(branch):

    compiler = find_compiler()
    # the configuration for GCL python extension
    config = {}
    config['name'] = '{}.{}.binding'.format(package_basename, branch)
    config['extra_link_args'] = ['-fPIC']
    config['extra_compile_args'] = []
    # important or get a symbol not found error, because class is
    # compiled with c++?
    config['language'] = 'c'
    config['libraries'] = ['m']

    # determine if swig needs to be called
    config['sources'] = [os.path.join(package_basename, branch, 'binding.pyx')]
    os.environ.setdefault('CC', compiler)
    if compiler_is_clang(compiler):
        # see https://github.com/lesgourg/class_public/issues/405
        os.environ.setdefault('OMPFLAG', '-Xclang -fopenmp')
        os.environ.setdefault('CCFLAG', os.environ.get('CFLAGS', ''))  # no -fPIC
        os.environ.setdefault('LDFLAG', '-fPIC -lomp')
        config['extra_link_args'] += ['-lomp']
    else:
        config['extra_link_args'] += ['-lgomp']
    if compiler in ['cc', 'icc']:
        # see https://github.com/lesgourg/class_public/issues/40
        config['libraries'] += ['irc', 'svml', 'imf']
        config['extra_link_args'] += ['-liomp5']

    return config


if __name__ == '__main__':

    setup(name=package_basename,
          version=version,
          author='Arnaud de Mattia, based on classylss by Nick Hand, Yu Feng',
          author_email='',
          description='Python binding of the CLASS CMB Boltzmann code',
          license='GPL3',
          url='http://github.com/adematti/pyclass',
          install_requires=['numpy', 'cython'],
          ext_modules=[Extension(**classy_extension_config(branch)) for branch in find_branches()],
          cmdclass={'build_ext': custom_build_ext,
                    'develop': custom_develop,
                    'clean': custom_clean},
          packages=find_packages())
