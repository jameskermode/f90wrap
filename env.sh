#! /bin/sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLATFORM=`python -c "from distutils.util import get_platform ; from distutils.sysconfig import get_python_version ; print '%s-%s' % ( get_platform(), get_python_version() )"`

export PYTHONPATH="$ROOT/build/lib.$PLATFORM:$PYTHONPATH"
