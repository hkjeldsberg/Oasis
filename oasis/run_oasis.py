#!/usr/bin/env python

import os
import sys

sys.path.append(os.getcwd())


def main():
    assert sys.argv[1] in ('NSfracStep', 'NSCoupled' 'NSfracStepMove')
    solver = sys.argv.pop(1)
    if solver == 'NSfracStep':
        pass

    elif solver == 'NSCoupled':
        pass

    elif solver == "NSfracStepMove":
        pass

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
