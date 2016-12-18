#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

from __future__ import print_function
import os
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install
import time

try:
    exec(open('systemml/project_info.py').read())
except IOError:
    print("Could not read project_info.py. Will use default values.", file=sys.stderr)
    BUILD_DATE_TIME = str(time.strftime("%Y%m%d.%H%M%S"))
    __project_artifact_id__ = 'systemml'
    __project_version__ = BUILD_DATE_TIME + '.dev0'
ARTIFACT_NAME = __project_artifact_id__
ARTIFACT_VERSION = __project_version__
ARTIFACT_VERSION_SHORT = ARTIFACT_VERSION.split("-")[0]

numpy_version = '1.8.2'
scipy_version = '0.15.1'
REQUIRED_PACKAGES = [
    'numpy >= %s' % numpy_version,
    'scipy >= %s' % scipy_version
]


python_dir = 'systemml'
java_dir='systemml-java'
java_dir_full_path = os.path.join(python_dir, java_dir)
PACKAGE_DATA = []
for path, subdirs, files in os.walk(java_dir_full_path):
    for name in files:
        PACKAGE_DATA = PACKAGE_DATA + [ os.path.join(path, name).replace('./', '') ]
cpp_dir='systemml-cpp'
cpp_dir_full_path = os.path.join(python_dir, cpp_dir)
for path, subdirs, files in os.walk(cpp_dir_full_path):
    for name in files:
        PACKAGE_DATA = PACKAGE_DATA + [ os.path.join(path, name).replace('./', '') ]
PACKAGE_DATA = PACKAGE_DATA + [os.path.join(python_dir, 'LICENSE'), os.path.join(python_dir, 'DISCLAIMER'), os.path.join(python_dir, 'NOTICE')]

java_home = os.environ.get('JAVA_HOME')
blas_type = os.environ.get('USE_BLAS') if os.environ.get('USE_BLAS') is not None else 'mkl' 
from sys import platform
if platform == "linux" or platform == "linux2":
    my_os = 'linux'
else:
    my_os = platform

def get_mkl_root():
    if os.environ.get('MKLROOT') is not None:
        return os.environ.get('MKLROOT')
    elif my_os == 'linux' or my_os == 'darwin':
        if os.path.isdir('/opt/intel/mkl'):
            return '/opt/intel/mkl'
    elif my_os == 'win32':
        # TODO: Return default MKL installation path
        return None
    return None

def get_openblas_root():
    if os.environ.get('OPENBLASROOT') is not None:
        return os.environ.get('OPENBLASROOT')
    elif my_os == 'linux' or my_os == 'darwin':
        if os.path.isdir('/opt/openblas'):
            return '/opt/openblas'
    elif my_os == 'win32':
        # TODO: Return default MKL installation path
        return None
    return None

      
def get_include_dirs(blas_root):
    ret = [ os.path.join(java_home, 'include'), 'systemml-cpp', os.path.join(blas_root, 'include') ]
    if my_os == 'linux':
        ret = ret + [ os.path.join(java_home, 'include', 'linux') ]
    elif my_os == 'win32':
        ret = ret + [ os.path.join(java_home, 'include', 'win32') ]
    # TODO: MacOSX
    return ''.join([ ' -I' + include_path for include_path in ret ])
        
def get_linker_flags(blas_root):
    if my_os == 'linux' and blas_type == 'mkl':
        import subprocess
        is_64_bit = '64' in subprocess.Popen(["getconf", "LONG_BIT"], stdout=subprocess.PIPE).communicate()[0]
        if is_64_bit:
            mkl_lib_path = '-L' + os.path.join(blas_root, 'lib', 'intel64') + ' -m64 -Wl,--no-as-needed'
        else:
            mkl_lib_path = '-L' + os.path.join(blas_root, 'lib', 'ia32') + ' -m32 -Wl,--no-as-needed'
        return ' -lmkl_rt -lpthread -lm -ldl ' + mkl_lib_path
    elif my_os == 'linux' and blas_type == 'openblas':
        return ' -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -L' + os.path.join(blas_root, 'lib')
    elif my_os == 'win32' and blas_type == 'mkl':
        mkl_lib_path = [ 'mkl_intel_c_dll.lib', 'mkl_intel_thread_dll.lib', 'mkl_core_dll.lib' ]
        is_64_bit = False if 'x86' in blas_root else True
        if is_64_bit:
            mkl_lib_path = [ os.path.join(blas_root, 'lib', 'intel64_win', l) for l in mkl_lib_path ]
        else:
            mkl_lib_path = [ os.path.join(blas_root, 'lib', 'ia32_win', l) for l in mkl_lib_path ]
        return ' -MD -LD ' + ' '.join(mkl_lib_path)

def get_other_flags():
    if my_os == 'linux':
        return ' -fopenmp -O3 -shared -fPIC'
    elif my_os == 'win32':
        return ' /openmp'
        
def compile_cpp():
    is_jdk_installed = (java_home is not None) and os.path.isdir(os.path.join(java_home, 'include'))
    if not is_jdk_installed:
        print('JAVA_HOME is not set. Skipping the building of systemml native library.')
        return
    blas_root = get_openblas_root() if blas_type == 'openblas' else get_mkl_root()
    if blas_root is None:
        print('Cannot find BLAS root directory. Skipping the building of systemml native library.')
        return
    if my_os == 'linux':
        cmd = 'g++ -o ' + os.path.join(os.path.expanduser('~'), 'libsystemml.so') + ' systemml-cpp/systemml.cpp '
    elif my_os == 'win32':
        cmd = 'cl systemml.cpp -Fe'  + os.path.join(os.path.expanduser('~'), 'systemml.dll') + ' '
    else:
        cmd = 'TODO'
    cmd = cmd + get_include_dirs(blas_root) + get_linker_flags(blas_root) + get_other_flags()
    print('\n==========================================================')
    should_compile = os.environ.get('COMPILE_NATIVE')
    should_compile = should_compile is not None and should_compile == '1'
    if should_compile:
        print('Executing following command to compile native systemml library:')
    else:
        print('Please use following command to compile native systemml library:')
    print(cmd)
    if should_compile:
        os.system(cmd.replace('systemml-cpp', os.path.join('systemml', 'systemml-cpp')))
    print('==========================================================\n')

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        compile_cpp()

setup(
    name=ARTIFACT_NAME,
    version=ARTIFACT_VERSION_SHORT,
    description='Apache SystemML is a distributed and declarative machine learning platform.',
    long_description='''

    Apache SystemML is an effort undergoing incubation at the Apache Software Foundation (ASF), sponsored by the Apache Incubator PMC.
    While incubation status is not necessarily a reflection of the completeness
    or stability of the code, it does indicate that the project has yet to be
    fully endorsed by the ASF.

    Apache SystemML provides declarative large-scale machine learning (ML) that aims at
    flexible specification of ML algorithms and automatic generation of hybrid runtime
    plans ranging from single-node, in-memory computations, to distributed computations on Apache Hadoop and Apache Spark.
    ''',
    url='http://systemml.apache.org/',
    author='Apache SystemML',
    author_email='dev@systemml.incubator.apache.org',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    package_data={
        'systemml-java': PACKAGE_DATA
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    cmdclass={ 'install': CustomInstallCommand },
    )
