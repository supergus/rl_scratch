"""This is an awesome module"""

__author__ = "Christopher Couch"
__license__ = "Strictly proprietary for Liveline Technologies, Inc."
__version__ = "2020-01"

# Includes
import os


# Class definition
class Data(object):
    """A special class..

    |
    Properties:

    - `prop_A`:  Foo.
    - `prop_B`: Foo.
    
    |
    Methods:

    - `meth_A`:  Foo.
    - `meth_B`: Foo. 
    """

    def __init__(self):
        return

    def __repr__(self):
        return 'This is a generic repr!'

    def method_1(self, required_pos_arg, *args, named_kw_arg=False, **kwargs):
        """This method does great things.
        
        Let's talk about them here.
        
        Arguments:
            required_pos_arg (int): Required positional argument
            
        Keyword Arguments:
            kw_arg_1 (bool): Optional keyword argument
            
        Returns:
            Some helpful things.
        """

        print('=' * 80)

        print('REQUIRED POSITIONAL ARGUMENTS:')
        print(f'required_pos_arg: {required_pos_arg}')

        print('\nOPTIONAL POSITIONAL ARGUMENTS:')
        for a in args:
            print(f'{a}')

        print('\nNAMED KEYWORD ARGUMENTS:')
        print(f'named_kw_arg: {named_kw_arg}')

        print('\nOPTIONAL KEYWORD ARGUMENTS:')
        for k in kwargs:
            print(f'keyword argument `{k}` is: {kwargs[k]}')

        return


def main():
    d = Data()
    d.method_1(42, 'foo', 'fighter', named_kw_arg=True, another_kwarg=3.14159, yet_another_kwarg='yet more')
    d.method_1(42, 'foo', 'fighter', yet_another_kwarg='yet more')


if __name__ == '__main__':
    fn = os.path.basename(__file__)
    print('\nRunning module `{}` directly...\n'.format(fn))
    main()
