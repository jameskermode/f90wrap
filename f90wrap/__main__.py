import argparse
import sys
from importlib import import_module

commands = {
        '--f90wrap' : 'f90wrap.scripts.main',
        '--f2py' : 'f90wrap.scripts.f2py_f90wrap',
        '--f2py-f90wrap' : 'f90wrap.scripts.f2py_f90wrap',
        '--f90doc' : 'f90wrap.scripts.f90doc',
        }

def main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and ('-h' in sys.argv or '--help' in sys.argv)):
        parser = argparse.ArgumentParser(description='f90wrap : F90 to Python interface generator with derived type support')
        parser.description = 'f90wrap tasks :\n' + '\n\t'.join(commands.keys())
        parser.add_argument('--f90wrap', action='store_true', default=False, help='main function of f90wrap')
        parser.add_argument('--f2py', '--f2py-f90wrap', action='store_true', default=False, help='f90wrap patched version of f2py')
        parser.add_argument('--f90doc', action='store_true', default=False, help='documentation generator for Fortran 90')
        parser.parse_args()
        if len(sys.argv) == 1 : parser.print_help()
    else :
        for job in commands :
            if job in sys.argv :
                sys.argv.remove(job)
                return getattr(import_module(commands[job]), 'main')()
        else :
            job = '--f90wrap'
            return getattr(import_module(commands[job]), 'main')()


if __name__ == "__main__":
    main()
