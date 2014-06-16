# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

class Struct:
    def __init__(self, content=None):
        if type(content) == dict:
            self.__dict__ = content

    def __getitem__(self, index):
        return self.__dict__[index]

    def __setitem__(self, index, value):
        self.__dict__[index] = value

    def __iter__(self):
        for i in self.__dict__:
            yield self[i]

    def keys(self):
        return self.__dict__.keys()

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        longest = max([len(i) for i in self.__dict__])
        return '\n'.join(['%s : ' % i.rjust(longest) + str(self[i]) for i in self.__dict__])

    def __repr__(self):
        return 'Struct( %s )' % str(self.__dict__)
        
