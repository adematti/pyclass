"""
Utilities for CLASS external files.
Taken from https://github.com/nickhand/classylss/blob/master/classylss/__init__.py.
"""


def get_external_files(**kwargs):
    """
    Return the path of external files required for running CLASS.
    They are installed to the package directory, in the ``external`` folder.
    """
    import os

    path = os.path.dirname(__file__)
    path = os.path.join(path, 'external')
    toret = {'hyrec_path': os.path.join(path, 'HyRec2020') + os.sep,
             'Galli_file': os.path.join(path, 'heating', 'Galli_et_al_2013.dat'),
              'sd_external_path': os.path.join(path, 'distortions') + os.sep,
              'sBBN file': os.path.join(path, 'bbn', 'sBBN_2017.dat')}
    for name in toret:
        if name in kwargs:
            toret[name] = _find_file(kwargs[name])
    return toret


def _find_file(filename):
    """Find the file path, first checking if it exists and then looking in the ``external`` directory."""
    import os
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'external', filename)

    if not os.path.exists(path):
        raise ValueError('Cannot locate file {}'.format(filename))

    return path


def load_ini(filename):
    """
    Read a CLASS ``.ini`` file, returning a dictionary of parameters.

    Parameters
    ----------
    filename : str
        The name of an existing parameter file to load, or one included as part of the CLASS source.

    Returns
    -------
    ini : dict
        The input parameters loaded from file.
    """
    # also look in data dir
    filename = _find_file(filename)

    params = {}
    with open(filename, 'r') as file:

        # loop over lines
        for lineno, line in enumerate(file):
            if not line: continue

            # skip any commented lines with #
            if '#' in line: line = line[line.index('#') + 1:]

            # must have an equals sign to be valid
            if "=" not in line: continue

            # extract key and value pairs
            fields = line.split('=')
            if len(fields) != 2:
                import warnings
                warnings.warn('Skipping line number {}: "{}"'.format(lineno, line))
                continue
            params[fields[0].strip()] = fields[1].strip()

    return params


def load_precision(filename):
    """
    Load a CLASS ``.pre`` file, returning a dictionary of parameters.

    Parameters
    ----------
    filename : str
        The name of an existing file to load, or one in the files included as part of the CLASS source.

    Returns
    -------
    pre : dict
        The precision parameters loaded from file.
    """
    return load_ini(filename)
