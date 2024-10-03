# cython: embedsignature=True
# cython: binding=True
import os
import functools
import inspect

cimport cython
import numpy as np
from numpy.core.numeric import normalize_axis_index
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset, strncpy, strdup
from libc.math cimport exp, sqrt

from .cclassy cimport *

from .utils import get_external_files

DEF _Mpc_over_m_ = 3.085677581282e22  #  /**< conversion factor from meters to megaparsecs */
#/* remark: CAMB uses 3.085678e22: good to know if you want to compare  with high accuracy */
DEF _Gyr_over_Mpc_ = 3.06601394e2     #  /**< conversion factor from megaparsecs to gigayears
          #       (c=1 units, Julian years of 365.25 days) */
DEF _c_ = 2.99792458e8                #  /**< c in m/s */
DEF _G_ = 6.67428e-11                 #  /**< Newton constant in m^3/Kg/s^2 */
DEF _eV_ = 1.602176487e-19            #  /**< 1 eV expressed in J */
DEF _PI_ = 3.1415926535897932384626433832795e0

# /* parameters entering in Stefan-Boltzmann constant sigma_B */
DEF _k_B_ = 1.3806504e-23
DEF _h_P_ = 6.62606896e-34

# NOTE: using here up to scipy accuracy; replace by e.g. CLASS accuracy?
DEF msun = 1.98847 * 1e30  # kg
# h^2 * kg/m^3
cdef float rho_crit_kgph_per_mph3 = 3.0 * (100. * 1e3 / _Mpc_over_m_)**2 / (8 * np.pi * _G_)
# h^2 * kg/m^3 / 10^10 msun / Mpc^3 = 10^10 Msun/h / (Mpc/h)^3
cdef float rho_crit_Msunph_per_Mpcph3 = rho_crit_kgph_per_mph3 / (10**10 * msun) * _Mpc_over_m_**3

DEF _MAX_NUMBER_OF_K_FILES_ = 30
DEF _MAXTITLESTRINGLENGTH_ = 8000
DEF _LINE_LENGTH_MAX_ = 1024

DEF NAN = float('NaN')


class ClassInputError(ValueError):
    r"""Exception raised for an issue with the input parameters."""

    def __init__(self, message, file_content):
        self.message = message
        self.file_content = file_content

    def __str__(self):
        return 'CLASS input error :{}\n{}'.format(self.message, self.file_content)


class ClassComputationError(ValueError):
    r"""Raised when CLASS could not compute the cosmology with the provided parameters."""


class ClassRuntimeError(ValueError):
    r"""Exception raised when accessing CLASS-computed quantities."""

    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return 'CLASS runtime error: {}'.format(self.message)


def val2str(val):
    """Turn ``val`` into string."""
    if (not isinstance(val, str)) and np.iterable(val):
        return ','.join([str(i) for i in val])
    return str(val).strip()


def joinstr(s, *args):
    r"""Separate ``args`` with commas if ``s`` is not empty."""
    if s: return ','.join((s,) + args)
    return ','.join(args)


def is_sequence(item):
    return isinstance(item, (tuple, list))


def _bcast_dtype(*args):
    r"""If input arrays are all float32, return float32; else float64."""
    toret = np.result_type(*(getattr(arg, 'dtype', None) for arg in args))
    if not np.issubdtype(toret, np.floating):
        toret = np.float64
    return toret


def flatarray(iargs=[0], dtype=np.float64):
    """Decorator that flattens input array(s) and reshapes the output in the same form."""
    def make_wrapper(func):
        sig = inspect.signature(func)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            self, args = ba.args[0], list(ba.args[1:])
            toret_dtype = _bcast_dtype(*[args[iarg] for iarg in iargs])
            input_dtype = dtype
            if input_dtype is None:
                input_dtype = toret_dtype
            shape = None
            for iarg in iargs:
                array = np.asarray(args[iarg], dtype=input_dtype)
                if shape is not None:
                    if array.shape != shape:
                        raise ValueError('input arrays must have same shape, found {}, {}'.format(shape, array.shape))
                else:
                    shape = array.shape
                args[iarg] = array.ravel()

            toret = func(self, *args, **ba.kwargs)

            def reshape(toret):
                toret = np.asarray(toret, dtype=toret_dtype)
                toret.shape = toret.shape[:-1] + shape
                return toret

            if isinstance(toret, dict):
                for key, value in toret.items():
                    toret[key] = reshape(value)
            else:
                toret = reshape(toret)

            return toret

        return wrapper

    return make_wrapper


def gridarray(iargs=[0], dtype=np.float64):
    r"""
    Decorator that shapes output as ``(x1.size, x2.size, ...)`` for input arrays ``(x1, x2, ...)``.
    Dimensions corresponding to scalar inputs are squeezed.
    """
    def make_wrapper(func):
        sig = inspect.signature(func)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            self, args = ba.args[0], list(ba.args[1:])
            toret_dtype = _bcast_dtype(*[args[iarg] for iarg in iargs])
            toret_shape = tuple()
            for iarg in iargs:
                array = np.asarray(args[iarg], dtype=dtype)
                toret_shape = toret_shape + array.shape
                args[iarg] = array.ravel()
            toret = np.asarray(func(self, *args, **ba.kwargs), dtype=toret_dtype)
            toret.shape = toret_shape
            return toret

        return wrapper

    return make_wrapper


def _compile_params(params):
    r"""Build up parameter dictionary: removes ``None``, enforce ``output`` (calculation being determined by :meth:`ClassEngine.compute`)."""
    params = {**params, **get_external_files(**params)}
    if 'verbose' in params:
        params.pop('verbose')
        verbose = params['verbose']
        if verbose:
            for section in ['input', 'background', 'thermodynamics', 'perturbations',
                            'primordial', 'transfer', 'fourier', 'harmonic', 'lensing', 'distortions']:
                name = '{}_verbose'.format(section)
                if name not in params: params[name] = 1
    for key in list(params.keys()):
        if params[key] is None: params.pop(key)
    params.setdefault('output', ['dTk', 'vTk', 'tCl', 'pCl', 'lCl', 'mPk', 'nCl'])
    params.setdefault('number_count_contributions', ['density', 'rsd', 'lensing'])  # calculation in Perturbations very expansive when asking for Harmonic
    number_count_contributions = params.get('number_count_contributions', [])
    if not number_count_contributions:
        try: params['output'].remove('nCl')
        except ValueError: pass
    if 'nCl' not in params['output']:
        params.pop('number_count_contributions', None)
    return params


cdef int index_symmetric_matrix(int i1, int i2, int N) nogil:
    r"""CLASS macros, definded in common.h, l. 76."""
    if i1 <= i2:
        return i2 + N*i1 - i1 * (i1 + 1) // 2
    return i1 + N * i2 - i2 * (i2 + 1) // 2


cdef int _build_file_content(params, file_content * fc) except -1:
    r"""Dump parameter dictionary ``params`` to file structure ``fc``."""
    parser_free(fc)
    fc.filename = <char*>malloc(sizeof(FileArg))
    strncpy(fc.filename, 'NOFILE', sizeof(FileArg))
    fc.size = len(params)
    fc.name = <FileArg*> malloc(sizeof(FileArg) * len(params))
    assert(fc.name != NULL)
    fc.value = <FileArg*> malloc(sizeof(FileArg) * len(params))
    assert(fc.value != NULL)
    fc.read = <short*> malloc(sizeof(short) * len(params))
    assert(fc.read != NULL)
    # fill parameter file
    for ii, kk in enumerate(params.keys()):
        dumcp = val2str(kk).encode()
        strncpy(fc.name[ii], dumcp[:sizeof(FileArg) - 1], sizeof(FileArg))
        dumcp = val2str(params[kk]).encode()
        strncpy(fc.value[ii], dumcp[:sizeof(FileArg) - 1], sizeof(FileArg))
        fc.read[ii] = _FALSE_
    return 0


cdef np.dtype _titles_to_dtype(char * titles, int remove_units=False):
    r"""Turn CLASS ``titles`` char array into ``dtype``."""
    tmp = (<bytes>titles).decode()
    names = tmp.split('\t')[:-1]
    number_of_titles = len(names)
    if remove_units:
        dtype = np.dtype([(str(name.split()[0]), np.float64) for name in names])
    else:
        dtype = np.dtype([(str(name), np.float64) for name in names])
    return dtype


def _build_task_dependency(tasks):
    r"""
    Fill the task list with all the needed modules.

    .. warning::

        the ordering of modules is obviously dependent on CLASS module order
        in the main.c file. This has to be updated in case of a change to
        this file.

    Parameters
    ----------
    tasks : list
        list of strings, containing initially only the last module required.
        For instance, to recover all the modules, the input should be ``['distortions']``.

    Returns
    -------
    tasks : list
        Complete task list.
    """
    if not is_sequence(tasks): tasks = [tasks]
    tasks = list(tasks)
    if 'distortions' in tasks:
        tasks.append('lensing')
    if 'lensing' in tasks:
        tasks.append('harmonic')
    if 'harmonic' in tasks:
        tasks.append('transfer')
    if 'transfer' in tasks:
        tasks.append('fourier')
    if 'fourier' in tasks:
        tasks.append('primordial')
    if 'primordial' in tasks:
        tasks.append('perturbations')
    if 'perturbations' in tasks:
        tasks.append('thermodynamics')
    if 'thermodynamics' in tasks:
        tasks.append('background')
    if len(tasks) != 0:
        tasks.append('input')
    return tasks


ctypedef struct ready_flags:
    int pr
    int ba
    int th
    int pt
    int tr
    int pm
    int hr
    int op
    int sd
    int le
    int fo
    int fc
    int ip


cdef class ClassEngine:
    r"""CLASS engine class, which initialises CLASS from an input set of parameters, and runs calculations with :meth:`compute`."""
    cdef precision pr
    cdef background ba
    cdef thermodynamics th
    cdef perturbations pt
    cdef transfer tr
    cdef primordial pm
    cdef harmonic hr
    cdef output op
    cdef distortions sd
    cdef lensing le
    cdef fourier fo
    cdef file_content fc
    cdef ready_flags ready
    cdef int l_scalar_max

    def __init__(self, object params={}):
        r"""
        Initialize ``ClassEngine``.

        Parameters
        ----------
        params : dict, optional
            Dictionary of input parameters, following naming conventions of ``explanatory.ini``.
        """
        params = _compile_params(params)
        _build_file_content(params, &self.fc)
        self.ready.fc = True
        self.compute('input')

    def __cinit__(self, *args, **kwargs):
        memset(&self.ready, 0, sizeof(self.ready))

    def get_params(self, return_type='dict'):
        r"""Return parameter dictionary as passed to CLASS."""
        if not self.ready.fc:
            toret = {}
        toret = {self.fc.name[i].decode(): self.fc.value[i].decode() for i in range(self.fc.size)}
        if return_type == 'dict':
            return toret
        return '\n'.join(['{} = {}'.format(name, value) for name, value in toret.items()])

    def __dealloc__(self):
        r"""Free C structures."""
        # print(self.ready.ba, self.ready.th, self.ready.pt, self.ready.pm, self.ready.fo, self.ready.tr, self.ready.hr, self.ready.le, self.ready.sd, self.ready.fc, 'dealloc')
        if self.ready.fc: parser_free(&self.fc)
        if self.ready.sd: distortions_free(&self.sd)
        if self.ready.le: lensing_free(&self.le)
        if self.ready.hr: harmonic_free(&self.hr)
        if self.ready.tr: transfer_free(&self.tr)
        if self.ready.fo: fourier_free(&self.fo)
        if self.ready.pm: primordial_free(&self.pm)
        if self.ready.pt: perturbations_free(&self.pt)
        if self.ready.th: thermodynamics_free(&self.th)
        if self.ready.ba: background_free(&self.ba)

    def compute(self, tasks):
        r"""
        The main function, which executes the 'init' methods for all the desired modules, in analogy to ``main.c``.

        Note
        ----
        For speed, ask for 'harmonic' first if desired.

        Parameters
        ----------
        tasks : list, string
            Calculation to perform, in the following list:
            ['input', 'background', 'thermodynamics', 'perturbations', 'primordial', 'fourier', 'transfer', 'harmonic', 'lensing', 'distortions']
        """
        cdef file_content * fc = &self.fc
        cdef ErrorMsg errmsg
        if not is_sequence(tasks): tasks = [tasks]
        input_tasks = list(tasks)
        tasks = _build_task_dependency(tasks)

        # --------------------------------------------------------------------
        # Check the presence for all CLASS modules in the list 'tasks'. If a
        # module is found in tasks, executure its '_init' method.
        # --------------------------------------------------------------------
        # The input module should raise a ClassRuntimeError, because
        # non-understood parameters asked to the wrapper is a problematic
        # situation.
        if 'input' in tasks and not self.ready.ip:
            environ_bak = dict(os.environ)
            try:
                os.environ['LC_NUMERIC'] = 'C'
                if input_read_from_file(&self.fc, &self.pr, &self.ba, &self.th,
                                        &self.pt, &self.tr, &self.pm, &self.hr,
                                        &self.fo, &self.le, &self.sd, &self.op,
                                        errmsg) == _FAILURE_:
                    raise ClassInputError(errmsg.decode(), self.get_params(return_type='str'))
            finally:
                os.environ.clear()
                os.environ.update(environ_bak)
            # This part is done to list all the unread parameters, for debugging
            unread_parameters = []
            for ii in range(fc.size):
                if fc.read[ii] == _FALSE_:
                    unread_parameters.append(fc.name[ii].decode())

            if unread_parameters:
                import warnings
                warnings.warn('Class did not read input parameter(s): {}'.format(', '.join(unread_parameters)))
            self.ready.ip = True
            self.l_scalar_max = self.pt.l_scalar_max

        def short(b):
            return _TRUE_ if b else _FALSE_

        compute_background = 'background' in tasks and not self.ready.ba
        compute_thermodynamics = 'thermodynamics' in tasks and not self.ready.th
        compute_lensing = 'lensing' in tasks and not self.ready.le

        #if compute_lensing:
        #    if self.le.has_lensed_cls == _FALSE_:
        #        self.pt.l_scalar_max = self.l_scalar_max + self.pr.delta_l_max
        #        self.ready.hr = False
        #        self.le.has_lensed_cls = _TRUE_

        compute_transfer = 'transfer' in tasks and not self.ready.tr
        compute_harmonic = 'harmonic' in tasks and not self.ready.hr
        compute_fourier = 'fourier' in tasks and not self.ready.fo
        compute_primordial = 'primordial' in tasks and not self.ready.pm
        compute_distortions = 'distortions' in tasks and not self.ready.sd
        compute_perturbations = 'perturbations' in tasks and not self.ready.pt
        compute_transfer |= compute_harmonic  # to avoid segfault if e.g. transfer then harmonic, as harmonic requires more sampling
        compute_fourier |= compute_harmonic
        compute_primordial |= compute_harmonic
        compute_perturbations |= compute_harmonic

        # This to avoid computing a very fine sampling if Harmonic is not required
        # But then, perturbations, primordial, fourier, transfer need to be recomputed with finer sampling when harmonic is required
        # So ask for harmonic first if it is desired!
        self.pt.has_cl_cmb_temperature = short(compute_harmonic or self.ready.hr)
        self.pt.has_cl_cmb_polarization = short(compute_harmonic or self.ready.hr)
        self.pt.has_cls = short(compute_harmonic or self.ready.hr)
        """
        # Already in params['number_count_contributions'] = ['density', 'rsd', 'gr']
        self.pt.has_cl_cmb_lensing_potential = self.le.has_lensed_cls
        self.pt.has_density_transfers = short(compute_transfer or self.ready.tr)
        self.pt.has_velocity_transfers = short(compute_transfer or self.ready.tr)
        self.pt.has_pk_matter = short(compute_fourier or self.ready.fo)
        """
        # to get theta_m, theta_cb, phi, psi, phi_plus_psi...
        # self.pt.has_cl_number_count = self.pt.has_nc_density = self.pt.has_nc_rsd = self.pt.has_nc_lens = _TRUE_
        # print(self.pt.has_cl_cmb_temperature, self.pt.has_cl_cmb_polarization, self.pt.has_cls, self.pt.has_cl_cmb_lensing_potential, self.pt.has_density_transfers, self.pt.has_velocity_transfers, self.pt.has_pk_matter)
        # print(self.pt.has_cl_number_count, self.pt.has_nc_rsd, self.pt.has_nc_lens)
        # The following list of computation is straightforward. If the '_init'
        # methods fail, call `struct_cleanup` and raise a ClassComputationError
        # with the error message from the faulty module of CLASS.
        # print(compute_background, compute_thermodynamics, compute_perturbations, compute_primordial, compute_fourier, compute_transfer, compute_harmonic, compute_lensing, compute_distortions)
        if compute_background:
            if background_init(&self.pr, &self.ba) == _FAILURE_:
                raise ClassComputationError(self.ba.error_message.decode())
            self.ready.ba = True

        if compute_thermodynamics:
            if thermodynamics_init(&self.pr, &self.ba, &self.th) == _FAILURE_:
                raise ClassComputationError(self.th.error_message.decode())
            self.ready.th = True

        if compute_perturbations:
            if perturbations_init(&self.pr, &self.ba, &self.th, &self.pt) == _FAILURE_:
                raise ClassComputationError(self.pt.error_message.decode())
            self.ready.pt = True

        if compute_primordial:
            if primordial_init(&self.pr, &self.pt, &self.pm) == _FAILURE_:
                raise ClassComputationError(self.pm.error_message.decode())
            self.ready.pm = True

        if compute_fourier:
            if fourier_init(&self.pr, &self.ba, &self.th,
                            &self.pt, &self.pm, &self.fo) == _FAILURE_:
                raise ClassComputationError(self.fo.error_message.decode())
            self.ready.fo = True

        if compute_transfer:
            if transfer_init(&self.pr, &self.ba, &self.th,
                             &self.pt, &self.fo, &self.tr) == _FAILURE_:
                raise ClassComputationError(self.tr.error_message.decode())
            self.ready.tr = True

        if compute_harmonic:
            if harmonic_init(&self.pr, &self.ba, &self.pt,
                             &self.pm, &self.fo, &self.tr,
                             &self.hr) == _FAILURE_:
                raise ClassComputationError(self.hr.error_message.decode())
            self.ready.hr = True

        if compute_lensing:
            if lensing_init(&(self.pr), &(self.pt), &(self.hr),
                            &(self.fo), &(self.le)) == _FAILURE_:
                raise ClassComputationError(self.le.error_message.decode())
            self.ready.le = True

        if compute_distortions:
            if distortions_init(&(self.pr), &(self.ba), &(self.th),
                                &(self.pt), &(self.pm), &(self.sd)) == _FAILURE_:
                raise ClassComputationError(self.sd.error_message.decode())
            self.ready.sd = True

        # At this point, the cosmological instance contains everything needed. The
        # following functions are only to output the desired numbers


cdef class Background:
    r"""
    Wrapper of the ``background`` module in CLASS.
    Distance unit is in :math:`\mathrm{Mpc}/h` (in CLASS: :math:`\mathrm{Mpc}`),
    density in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}` (in CLASS: :math:`\mathrm{Mpc}^{-2}`).
    """
    cdef ClassEngine engine
    cdef background * ba
    cdef readonly double _RH0_
    cdef readonly double Omega0_pncdm_tot
    cdef readonly np.ndarray Omega0_pncdm

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Background`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('background')
        self.ba = &self.engine.ba
        self._RH0_ = rho_crit_Msunph_per_Mpcph3 / self.ba.H0**2
        self.Omega0_pncdm_tot = self.Omega_pncdm_tot(0.0)
        self.Omega0_pncdm = self.Omega_pncdm(0.0, species=None)

    property Omega0_b:
        r"""Current density parameter of baryons :math:`\Omega_{b,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_b

    property Omega0_g:
        r"""Current density parameter of photons :math:`\Omega_{g,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_g

    property Omega0_cdm:
        r"""Current density parameter of cold dark matter :math:`\Omega_{cdm,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_cdm

    property Omega0_Lambda:
        r"""Current density parameter of cosmological constant :math:`\Omega_{\Lambda,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_lambda

    property Omega0_fld:
        r"""Current density parameter of dark energy (fluid) :math:`\Omega_{fld, 0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_fld

    property Omega0_de:
        r"""Current density parameter of dark energy (fluid + cosmological constant) :math:`\Omega_{de, 0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_fld + self.ba.Omega0_lambda

    property w0_fld:
        r"""Fluid equation of state parameter :math:`w_{0,\mathrm{fld}}`, unitless."""
        def __get__(self):
            return self.ba.w0_fld

    property wa_fld:
        r"""Fluid equation of state derivative :math:`w_{a,\mathrm{fld}}`, unitless."""
        def __get__(self):
            return self.ba.wa_fld

    property cs2_fld:
        r"""The sound speed defined in the frame comoving with the fluid :math:`c_{s}^{2}`, unitless."""
        def __get__(self):
            return self.ba.cs2_fld

    property Omega0_k:
        r"""Current density parameter of curvaturve :math:`\Omega_{k,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_k

    property K:
        r"""Curvature parameter, in :math:`(h/\mathrm{Mpc})^{2}`."""
        def __get__(self):
            return self.ba.K / self.h**2

    property Omega0_dcdm:
        r"""Current density parameter of decaying cold dark matter :math:`\Omega_{dcdm,0}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_dcdm

    property Omega0_ncdm:
        r"""Current density parameter of distinguishable (massive) neutrinos for each species as an array :math:`\Omega_{0, ncdm}`, unitless."""
        def __get__(self):
            return np.array([self.ba.Omega0_ncdm[i] for i in range(self.N_ncdm)], dtype=np.float64)

    property Omega0_ncdm_tot:
        r"""Current total density parameter of all distinguishable (massive) neutrinos, unitless."""
        def __get__(self):
            return self.ba.Omega0_ncdm_tot

    property Omega0_ur:
        r"""Current density parameter of ultra-relativistic (massless) neutrinos :math:`\Omega_{0,\nu_r}`, unitless."""
        def __get__(self):
            return self.ba.Omega0_ur

    property Omega0_r:
        r"""Current density parameter of radiation :math:`\Omega_{0,r}`, unitless.
        This is equal to:

        .. math::

            \Omega_{0,r} = \Omega_{0,g} + \Omega_{0,\nu_r} + \Omega_{0,pncdm}.
        """
        def __get__(self):
            #return self.ba.Omega0_g + self.ba.Omega0_ur + self.Omega0_pncdm_tot
            return self.ba.Omega0_r

    property Omega0_m:
        r"""
        The sum of density parameters for all non-relativistic components :math:`\Omega_{0,m}`, unitless.
        This is equal to:

        .. math::
            \Omega_{0,m} = \Omega_{0,b} + \Omega_{0,cdm} + \Omega_{0,ncdm} + \Omega_{0,dcdm} - \Omega_{0,pncdm}.
        """
        def __get__(self):
            #return self.ba.Omega0_b + self.ba.Omega0_cdm + self.ba.Omega0_ncdm_tot + self.ba.Omega0_dcdm - self.Omega0_pncdm_tot
            return self.ba.Omega0_m

    property N_eff:
        r"""Effective number of relativistic species, summed over ultra-relativistic and ncdm species."""
        def __get__(self):
            return self.ba.Neff

    property N_ur:
        r"""
        The number of ultra-relativistic species.

        This is equal to:

        .. math::

            N_{ur} = \Omega_{0,ur} / (7/8 (4/11)^{4/3} \Omega_{0,g}).
        """
        def __get__(self):
            return self.Omega0_ur / (7. / 8. * (4. / 11)**(4. / 3.) * self.Omega0_g)

    property N_ncdm:
        r"""The number of distinguishable ncdm (massive neutrino) species."""
        def __get__(self):
            return self.ba.N_ncdm

    property m_ncdm:
        r"""The masses of the distinguishable ncdm (massive neutrino) species, in :math:`\mathrm{eV}`."""
        def __get__(self):
            return np.array([self.ba.m_ncdm_in_eV[i] for i in range(self.N_ncdm)], dtype=np.float64)

    property m_ncdm_tot:
        r"""The sum of masses of the distinguishable ncdm (massive neutrino) species, in :math:`\mathrm{eV}`."""
        def __get__(self):
            return self.m_ncdm.sum()

    property age:
        r"""The current age of the Universe, in :math:`\mathrm{Gy}`."""
        def __get__(self):
            return self.ba.age

    property h:
        r"""The dimensionless Hubble parameter, unitless."""
        def __get__(self):
            return self.ba.h

    property H0:
        r"""The Hubble parameter, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        def __get__(self):
            return self.ba.h * 100

    property T0_cmb:
        r"""The current CMB temperature, in :math:`K`."""
        def __get__(self):
            return self.ba.T_cmb

    property T0_ncdm:
        r"""The current ncdm temperature for each species as an array, in :math:`K`."""
        def __get__(self):
            T = np.array([self.ba.T_ncdm[i] for i in range(self.N_ncdm)], dtype=np.float64)
            return T * self.ba.T_cmb # from units of photon temp to K

    @flatarray()
    def T_cmb(self, z):
        r"""The CMB temperature, in :math:`K`."""
        return self.T0_cmb * (1 + z)

    @flatarray()
    def T_ncdm(self, z, species=None):
        r"""
        The ncdm temperature (massive neutrinos), in :math:`K`.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is an index between 0 and N_ncdm (or a list of such indices), return temperature for this species.
        """
        return self.T0_ncdm[(species if species is not None else Ellipsis), None] * (1 + z)

    @cython.boundscheck(False)
    def _get_z(self, double[:] z, int column, int has=1, double default=0.):
        r"""Internal function to compute the background module at a specific redshift and return the ``column`` value."""
        # has specifies whether value is computed, else return default
        # Not an issue for z=0 values, as e.g. has_fld = ba.Omega0_fld != 0. (see background.c/background_indices)
        # Generate a new output array of the correct shape by broadcasting input arrays together
        cdef double [:] toret = np.empty_like(z)
        if not has:
            toret[:] = default
            return np.asarray(toret)

        cdef int last_index #junk
        cdef double [:] pvecback = np.empty(self.ba.bg_size, dtype=np.float64)
        cdef int iz, z_size = z.size

        with nogil:
            for iz in range(z_size):
                #if background_tau_of_z(self.ba, z[iz], &tau) == _FAILURE_:
                #    toret[iz] = NAN
                #elif background_at_tau(self.ba, tau, long_info, inter_normal, &last_index, &pvecback[0]) == _FAILURE_:
                #    toret[iz] = NAN
                if background_at_z(self.ba, z[iz], long_info, inter_normal, &last_index, &pvecback[0]) == _FAILURE_:
                    toret[iz] = NAN
                else:
                    toret[iz] = pvecback[column]
        return np.asarray(toret)

    @flatarray()
    def rho_g(self, z):
        r"""Comoving density of photons :math:`\rho_{g}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        .. note::

            In CLASS, :math:`\rho_{\mathrm{CLASS}} = 8 \pi G \rho_{\mathrm{physical}} / 3 c^{2}`.
        """
        return self._get_z(z, self.ba.index_bg_rho_g) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_b(self, z):
        r"""Comoving density of baryons :math:`\rho_{b}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_rho_b) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_m(self, z):
        r"""Comoving density of matter :math:`\rho_{m}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_Omega_m) * self.rho_crit(z)

    @flatarray()
    def rho_r(self, z):
        r"""Comoving density of radiation :math:`\rho_{r}`, including photons and relativistic part of massive and massless neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_Omega_r) * self.rho_crit(z)

    @flatarray()
    def rho_cdm(self, z):
        r"""Comoving density of cold dark matter :math:`\rho_{cdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_rho_cdm, self.ba.has_cdm) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_ur(self, z):
        r"""Comoving density of massless neutrinos :math:`\rho_{ur}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_rho_ur, self.ba.has_ur) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_dcdm(self, z):
        r"""Comoving density of decaying cold dark matter :math:`\rho_{dcdm}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self._get_z(z, self.ba.index_bg_rho_dcdm, self.ba.has_dcdm) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_ncdm(self, z, species=None):
        r"""
        Comoving density of non-relativistic part of massive neutrinos for each species, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is an index between 0 and N_ncdm (or a list of such indices), return density for this species.
        """
        if species is None:
            species = list(range(self.N_ncdm))

        if is_sequence(species):
            return np.array([self.rho_ncdm(z, species=s) for s in species]).reshape((len(species), len(z)))

        species = normalize_axis_index(species, self.N_ncdm)
        return self._get_z(z, self.ba.index_bg_rho_ncdm1 + species, self.ba.has_ncdm) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_ncdm_tot(self, z):
        r"""Total comoving density of non-relativistic part of massive neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return np.sum(self.rho_ncdm(z, species=None), axis=0)

    @flatarray()
    def rho_crit(self, z):
        r"""
        Comoving critical density excluding curvature :math:`\rho_{c}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{crit}}(z) = \frac{3 H(z)^{2}}{8 \pi G}.
        """
        return self._get_z(z, self.ba.index_bg_rho_crit) * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_k(self, z):
        r"""
        Comoving density of curvature :math:`\rho_{k}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined such that:

        .. math::

            \rho_{\mathrm{crit}} = \rho_\mathrm{tot} + \rho_k
        """
        return -self.ba.K * (1. + z)**2 * self._RH0_ / (1 + z)**3

    @flatarray()
    def rho_tot(self, z):
        r"""Comoving total density :math:`\rho_{\mathrm{tot}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return self.rho_crit(z) - self.rho_k(z)

    @flatarray()
    def rho_fld(self, z):
        r"""Comoving density of dark energy fluid :math:`\rho_{\mathrm{fld}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        if self.ba.has_fld:
            return self._get_z(z, self.ba.index_bg_rho_fld, self.ba.has_fld) * self._RH0_ / (1 + z)**3
        # return zeros of the right shape
        return self._get_z(z, self.ba.index_bg_a) * 0.0

    @flatarray()
    def rho_Lambda(self, z):
        r"""Comoving density of cosmological constant :math:`\rho_{\Lambda}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        if self.ba.has_lambda:
            return self._get_z(z, self.ba.index_bg_rho_lambda, self.ba.has_lambda) * self._RH0_ / (1 + z)**3
        # return zeros of the right shape
        return self._get_z(z, self.ba.index_bg_a) * 0.0

    @flatarray()
    def rho_de(self, z):
        r"""Total comoving density of dark energy :math:`\rho_{\mathrm{de}}`, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.

        This is defined as:

        .. math::

              \rho_{\mathrm{de}}(z) = \rho_{\mathrm{fld}}(z) + \rho_{\mathrm{\Lambda}}(z).
        """
        return self.rho_fld(z) + self.rho_Lambda(z)

    @flatarray()
    def p_ncdm(self, z, species=None):
        r"""
        Comoving pressure of non-relativistic part of massive neutrinos for each species, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is an index between 0 and N_ncdm (or a list of such indices), return pressure for this species.
        """
        if species is None:
            species = list(range(self.N_ncdm))

        if is_sequence(species):
            return np.array([self.p_ncdm(z, species=s) for s in species]).reshape((len(species), len(z)))

        species = normalize_axis_index(species, self.N_ncdm)
        return self._get_z(z, self.ba.index_bg_p_ncdm1 + species, self.ba.has_ncdm) * self._RH0_ / (1 + z)**3

    @flatarray()
    def p_ncdm_tot(self, z):
        r"""Total comoving pressure of non-relativistic part of massive neutrinos, in :math:`10^{10} M_{\odot}/h / (\mathrm{Mpc}/h)^{3}`."""
        return np.sum(self.p_ncdm(z, species=None), axis=0)

    @flatarray()
    def Omega_r(self, z):
        r"""Density parameter of radiation, including photons and relativistic part of massive and massless neutrinos, unitless."""
        return self.rho_r(z) / self.rho_crit(z)

    @flatarray()
    def Omega_m(self, z):
        r"""
        Density parameter of non-relativistic (matter-like) component, including
        non-relativistic part of massive neutrino, unitless.
        """
        return self.rho_m(z) / self.rho_crit(z)

    @flatarray()
    def Omega_g(self, z):
        r"""Density parameter of photons, unitless."""
        return self.rho_g(z) / self.rho_crit(z)

    @flatarray()
    def Omega_b(self, z):
        r"""Density parameter of baryons, unitless."""
        return self.rho_b(z) / self.rho_crit(z)

    @flatarray()
    def Omega_cdm(self, z):
        r"""Density parameter of cold dark matter, unitless."""
        return self.rho_cdm(z) / self.rho_crit(z)

    @flatarray()
    def Omega_k(self, z):
        r"""Density parameter of curvature, unitless."""
        return 1 - self.rho_tot(z) / self.rho_crit(z)

    @flatarray()
    def Omega_ur(self, z):
        r"""Density parameter of massless neutrinos, unitless."""
        return self.rho_ur(z) / self.rho_crit(z)

    @flatarray()
    def Omega_dcdm(self, z):
        r"""Density parameter of decaying cold dark matter, unitless."""
        return self.rho_dcdm(z) / self.rho_crit(z)

    @flatarray()
    def Omega_ncdm(self, z, species=None):
        r"""
        Density parameter of massive neutrinos, unitless.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is an index between 0 and N_ncdm (or a list of such indices), return density for this species.
        """
        return self.rho_ncdm(z, species=species) / self.rho_crit(z)

    @flatarray()
    def Omega_ncdm_tot(self, z):
        r"""Total density parameter of massive neutrinos, unitless."""
        return self.rho_ncdm_tot(z) / self.rho_crit(z)

    @flatarray()
    def Omega_pncdm(self, z, species=None):
        r"""
        Density parameter of pressure of non-relativistic part of massive neutrinos, unitless.
        If ``species`` is ``None`` returned shape is (N_ncdm,) if ``z`` is a scalar, else (N_ncdm, len(z)).
        Else if ``species`` is between 0 and N_ncdm, return density for this species.
        """
        return 3 * self.p_ncdm(z, species=species) / self.rho_crit(z)

    @flatarray()
    def Omega_pncdm_tot(self, z):
        r"""Total density parameter of pressure of non-relativistic part of massive neutrinos, unitless."""
        return 3 * self.p_ncdm_tot(z) / self.rho_crit(z)

    @flatarray()
    def Omega_fld(self, z):
        r"""Density parameter of dark energy (fluid), unitless."""
        return self.rho_fld(z) / self.rho_crit(z)

    @flatarray()
    def Omega_Lambda(self, z):
        r"""Density of cosmological constant, unitless."""
        return self.rho_Lambda(z) / self.rho_crit(z)

    @flatarray()
    def Omega_de(self, z):
        r"""Total density of dark energy (fluid + cosmological constant), unitless."""
        return self.rho_de(z) / self.rho_crit(z)

    @flatarray()
    def time(self, z):
        r"""Proper time (age of universe), in :math:`\mathrm{Gy}`."""
        return self._get_z(z, self.ba.index_bg_time) / _Gyr_over_Mpc_

    @flatarray()
    def comoving_radial_distance(self, z):
        r"""
        Comoving radial distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 15 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_C(z)`.
        """
        return self._get_z(z, self.ba.index_bg_conf_distance) * self.ba.h

    @flatarray()
    def conformal_time(self, z):
        r"""Conformal time, in :math:`\mathrm{Gy}`."""
        return (self.ba.conformal_age - self._get_z(z, self.ba.index_bg_conf_distance)) / _Gyr_over_Mpc_

    @flatarray()
    def hubble_function(self, z):
        r"""Hubble function ``ba.index_bg_H``, in :math:`\mathrm{km}/\mathrm{s}/\mathrm{Mpc}`."""
        return self._get_z(z, self.ba.index_bg_H) * _c_ / 1e3

    @flatarray()
    def hubble_function_prime(self, z):
        r"""
        Derivative of Hubble function: :math:`dH/d\tau`, where :math:`d\tau/da = 1 / (a^{2} H)`, in :math:`\mathrm{km}/\mathrm{s}`.

        Users should use :func:`efunc_prime` instead.
        """
        return self._get_z(z, self.ba.index_bg_H_prime) * _c_ / 1e3

    @flatarray()
    def efunc(self, z):
        r"""Function giving :math:`E(z)`, where the Hubble parameter is defined as :math:`H(z) = H_{0} E(z)`, unitless."""
        return self._get_z(z, self.ba.index_bg_H) / self.ba.H0

    @flatarray()
    def efunc_prime(self, z):
        r"""Function giving :math:`dE(z) / da`, unitless."""
        dtau_da = (1 + z)**2 / self.hubble_function(z)
        return self.hubble_function_prime(z) / self.ba.H0 * dtau_da

    @flatarray()
    def luminosity_distance(self, z):
        r"""
        Luminosity distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 21 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{L}(z)`.
        """
        return self._get_z(z, self.ba.index_bg_lum_distance) * self.ba.h

    @flatarray()
    def angular_diameter_distance(self, z):
        r"""
        Proper angular diameter distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 18 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{A}(z)`.
        """
        return self._get_z(z, self.ba.index_bg_ang_distance) * self.ba.h

    @flatarray()
    def comoving_angular_distance(self, z):
        r"""
        Comoving angular distance, in :math:`\mathrm{Mpc}/h`.

        See eq. 16 of `astro-ph/9905116 <https://arxiv.org/abs/astro-ph/9905116>`_ for :math:`D_{M}(z)`.
        """
        return self.angular_diameter_distance(z) * (1. + z)

    @flatarray(iargs=[0, 1])
    def angular_diameter_distance_2(self, z1, z2):
        r"""
        Angular diameter distance of object at :math:`z_{2}` as seen by observer at :math:`z_{1}`,
        that is, :math:`S_{K}((\chi(z_{2}) - \chi(z_{1})) \sqrt{|K|}) / \sqrt{|K|} / (1 + z_{2})`,
        where :math:`S_{K}` is the identity if :math:`K = 0`, :math:`\sin` if :math:`K < 0`
        and :math:`\sinh` if :math:`K > 0`.
        """
        if np.any(z2 < z1):
            import warnings
            warnings.warn(f"Second redshift(s) z2 ({z2}) is less than first redshift(s) z1 ({z1}).")
        chi1, chi2 = self.comoving_radial_distance(z1), self.comoving_radial_distance(z2)
        K = self.K
        if K == 0:
            return (chi2 - chi1) / (1 + z2)
        elif K > 0:
            return np.sin(np.sqrt(K) * (chi2 - chi1)) / np.sqrt(K) / (1 + z2)
        return np.sinh(np.sqrt(-K) * (chi2 - chi1)) / np.sqrt(-K) / (1 + z2)

    @flatarray()
    def growth_factor(self, z):
        r"""
        Return the scale invariant growth factor :math:`D(a)` for cold dark matter perturbations, unitless.

        This is the quantity defined by CLASS as ``index_bg_D`` in the background module, i.e. is the solution of:

        .. math::

            D^{\prime\prime}(\tau) = - a H(\tau) + \frac{3}{2} a^{2} \rho_{m} D(\tau)

        No ncdm in $\rho_{m} = \rho_{b} + \rho_{cdm} + \rho_{idmdr}, all species in $H$.
        """
        return self._get_z(z, self.ba.index_bg_D)

    @flatarray()
    def growth_rate(self, z):
        r"""
        Return the scale invariant growth rate :math:`d\mathrm{ln}D/d\mathrm{ln}a` for cold dark matter perturbations.

        This is the quantity defined by CLASS as ``index_bg_f`` in the background module.
        """
        return self._get_z(z, self.ba.index_bg_f)

    def table(self):
        r"""
        Return background table.

        Returns
        -------
        data : numpy.ndarray
            Structured array containing background data.
        """
        cdef char titles[_MAXTITLESTRINGLENGTH_]
        memset(titles, 0, _MAXTITLESTRINGLENGTH_)

        if background_output_titles(self.ba, titles) == _FAILURE_:
            raise ClassRuntimeError(self.ba.error_message.decode())

        dtype = _titles_to_dtype(titles)

        cdef np.ndarray data = np.empty(self.ba.bt_size, dtype=dtype)

        if background_output_data(self.ba, len(dtype.fields), <double*>data.data) == _FAILURE_:
            raise ClassRuntimeError(self.ba.error_message.decode())

        return data


cdef class Thermodynamics:

    r"""Wrapper of the ``thermodynamics` module in CLASS."""

    cdef ClassEngine engine
    cdef thermodynamics * th
    cdef background * ba

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Thermodynamics`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('thermodynamics')
        self.th = &self.engine.th
        self.ba = &self.engine.ba

    property z_drag:
        r"""Baryon drag redshift, unitless."""
        def __get__(self):
            return self.th.z_d

    property rs_drag:
        r"""Comoving sound horizon at the baryon drag epoch, in :math:`\mathrm{Mpc}/h`."""
        def __get__(self):
            return self.th.rs_d * self.ba.h

    property tau_reio:
        r"""Reionization optical depth, unitless."""
        def __get__(self):
            return self.th.tau_reio

    property z_reio:
        r"""Reionization redshift, unitless."""
        def __get__(self):
            return self.th.z_reio

    property tau_rec:
        r"""Optical depth at recombination, unitless."""
        def __get__(self):
            return self.th.tau_rec

    property z_rec:
        r"""Recombination redshift (at which the visibility reaches its maximum), unitless."""
        def __get__(self):
            return self.th.z_rec

    property rs_rec:
        r"""Comoving sound horizon at recombination, in :math:`\mathrm{Mpc}/h`."""
        def __get__(self):
            return self.th.rs_rec * self.ba.h

    property z_star:
        r"""Redshift of the last scattering surface (at which photon optical depth crosses one), unitless."""
        def __get__(self):
            return self.th.z_star

    property rs_star:
        r"""Comoving sound horizon at the last scattering surface, in :math:`\mathrm{Mpc}/h`."""
        def __get__(self):
            return self.th.rs_star * self.ba.h

    property theta_rec:
        r"""Sound horizon angle at recombination, equal to :math:`r_s(z_\mathrm{rec}) / D_A(z_\mathrm{rec})`, in radians."""
        def __get__(self):
            return self.th.rs_rec / self.th.da_rec / (1. + self.th.z_rec)

    property theta_star:
        r"""Sound horizon angle at the last scattering surface, equal to :math:`r_s(z_\mathrm{\star}) / D_A(z_\mathrm{\star})`, in radians."""
        def __get__(self):
            return self.th.rs_star / self.th.da_star / (1. + self.th.z_star)

    property YHe:
        r"""
        Helium mass fraction :math:`\rho_{He}/(\rho_{H} + \rho_{He})`, unitless.
        Close but not exactly equal to the density fraction :math:`4 n_{He}/(n_{H} + 4 n_{He})`.
        """
        def __get__(self):
            return self.th.YHe

    def table(self):
        r"""
        Return thermodynamics table.

        Returns
        -------
        data : numpy.ndarray
            Structured array containing thermodynamics data.
        """
        cdef char titles[_MAXTITLESTRINGLENGTH_]
        memset(titles, 0, _MAXTITLESTRINGLENGTH_)

        if thermodynamics_output_titles(self.ba, self.th, titles) == _FAILURE_:
            raise ClassRuntimeError(self.th.error_message.decode())

        dtype = _titles_to_dtype(titles)

        cdef np.ndarray data = np.empty(self.th.tt_size, dtype=dtype)

        if thermodynamics_output_data(self.ba, self.th, len(dtype.fields), <double*>data.data) == _FAILURE_:
            raise ClassRuntimeError(self.th.error_message.decode())

        return data


cdef class Primordial:

    r"""Wrapper of the ``primordial` module in CLASS."""

    cdef ClassEngine engine
    cdef perturbations * pt
    cdef primordial * pm
    cdef background * ba

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Primordial`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('transfer')
        self.pt = &self.engine.pt
        self.ba = &self.engine.ba
        self.pm = &self.engine.pm

    property A_s:
        r"""Scalar amplitude of the primordial power spectrum at :math:`k_\mathrm{pivot}`, unitless."""
        def __get__(self):
            return self.pm.A_s

    property ln_1e10_A_s:
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        def __get__(self):
            return np.log(1e10 * self.A_s)

    property n_s:
        r"""Power-law scalar index i.e. tilt of the primordial scalar power spectrum, unitless."""
        def __get__(self):
            return self.pm.n_s

    property alpha_s:
        r"""Running of the scalar spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        def __get__(self):
            return self.pm.alpha_s

    property beta_s:
        r"""Running of the running of the scalar spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        def __get__(self):
            return self.pm.beta_s

    property r:
        r"""Tensor-to-scalar power spectrum ratio at :math:`k_\mathrm{pivot}`, unitless."""
        def __get__(self):
            return self.pm.r

    property n_t:
        r"""Power-law tensor index i.e. tilt of the tensor primordial power spectrum, unitless."""
        def __get__(self):
            return self.pm.n_t

    property alpha_t:
        r"""Running of the tensor spectral index at :math:`k_\mathrm{pivot}`, unitless."""
        def __get__(self):
            return self.pm.alpha_t

    property k_pivot:
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        def __get__(self):
            return self.pm.k_pivot / self.ba.h

    @flatarray()
    @cython.boundscheck(False)
    def pk_k(self, double[:] k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, generated by inflation, in :math:`(\mathrm{Mpc}/h)^{3}`.
        For scalar perturbations this is e.g. defined as:

        .. math::

            \mathcal{P_R}(k) = A_s \left (\frac{k}{k_\mathrm{pivot}} \right )^{n_s - 1 + 1/2 \alpha_s \ln(k/k_\mathrm{pivot}) + 1/6 \beta_s \ln(k/k_\mathrm{pivot})^2}

        See also: eq. 2 of `this reference <https://arxiv.org/abs/1303.5076>`_.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        pk : numpy.ndarray, dict
            The primordial power spectrum if only one type of initial conditions (typically adiabatic),
            else dictionary of primordial power spectra corresponding to the tuples of initial conditions.
        """
        # generate a new output array of the correct shape by broadcasting input arrays together
        has_mode, index = Perturbations.get_index_md(self.pt, mode)
        cdef int index_md = index
        cdef double *primordial_pk = <double*> malloc(self.pt.ic_size[index_md] * sizeof(double))
        ic_keys = Perturbations.get_ic_keys(self.pt, index_md)
        ic_num = len(ic_keys)
        ic_ic_num = self.pm.ic_ic_size[index_md]
        cdef double[:, :] data = np.empty((k.size, ic_ic_num), dtype=np.float64)
        cdef int k_size = k.size
        cdef int ik

        if has_mode:
            with nogil:
                for ik in range(k_size):
                    if k[ik] == 0: # forcefully set k == 0 to zero.
                        data[ik, :] = 0.
                    elif primordial_spectrum_at_k(self.pm, index_md, linear, k[ik] * self.ba.h, &(data[ik, 0])) == _FAILURE_:
                        data[ik, :] = NAN
            # linear: all pks (contrary to cross-terms in logarithmic mode)
            data = np.asarray(data) * self.ba.h**3
        else:
            data[:, :] = 0.

        if len(ic_keys) > 1:
            toret = {}
            for index_ic1 in range(ic_num):
                for index_ic2 in range(index_ic1, ic_num):
                    ic_key = '{}_{}'.format(ic_keys[index_ic1], ic_keys[index_ic2])
                    index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic2, ic_num)
                    toret[ic_key] = np.asarray(data[:, index_ic1_ic2])
        else:
            toret = np.asarray(data[:, 0])
        return toret

    def table(self):
        r"""
        Return primordial table.

        Returns
        -------
        data : numpy.ndarray
            Structured array containing primordial data.
        """
        cdef char titles[_MAXTITLESTRINGLENGTH_]
        memset(titles, 0, _MAXTITLESTRINGLENGTH_)

        if primordial_output_titles(self.pt, self.pm, titles) == _FAILURE_:
            raise ClassRuntimeError(self.pm.error_message.decode())

        dtype = _titles_to_dtype(titles)

        cdef np.ndarray data = np.empty(self.pm.lnk_size, dtype=dtype)

        if primordial_output_data(self.pt, self.pm, len(dtype.fields), <double*>data.data) == _FAILURE_:
            raise ClassRuntimeError(self.pm.error_message.decode())

        #data['k [1/Mpc]'] /= self.ba.h
        return data


cdef class Perturbations:

    r"""Wrapper of the ``perturbations` module in CLASS."""

    cdef ClassEngine engine
    cdef perturbations * pt
    cdef background * ba

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Perturbations`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('transfer') # to actually get perturbations...
        self.pt = &self.engine.pt
        self.ba = &self.engine.ba

    property k_max_for_pk:
        r"""The input parameter specifying the maximum ``k`` value to compute spectra for, in :math:`h \mathrm{Mpc}^{-1}`."""
        def __get__(self):
            return self.pt.k_max_for_pk / self.ba.h

    property z_max_for_pk:
        r"""The input parameter specifying the maximum redshift measured for power spectra, unitless."""
        def __get__(self):
            return self.pt.z_max_pk

    property gauge:
        r"""The gauge name as a string, either 'newtonian' or 'synchronous'."""
        def __get__(self):
            if self.pt.gauge == newtonian:
                return 'newtonian'
            if self.pt.gauge == synchronous:
                return 'synchronous'
            raise ValueError('gauge value not understood')

    property has_scalars:
        r"""Whether scalar modes have been computed."""
        def __get__(self):
            return self.pt.has_scalars == _TRUE_

    property has_vectors:
        r"""Whether vector modes have been computed."""
        def __get__(self):
            return self.pt.has_vectors == _TRUE_

    property has_tensors:
        r"""Whether tensor modes have been computed."""
        def __get__(self):
            return self.pt.has_tensors == _TRUE_

    def table(self, mode='scalar'):
        r"""
        Return scalar, vector and/or tensor perturbations as arrays for requested k-values.

        .. note::

            You need to specify 'k_output_values' in input parameters.

        Parameters
        ----------
        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        perturbations : list of dict
            List of length 'k_output_values' of dictionaries containing perturbations.
        """
        cdef char titles[_MAXTITLESTRINGLENGTH_]
        cdef double * data[_MAX_NUMBER_OF_K_FILES_]
        cdef double[:, :] datak
        modes = ['scalar', 'vector', 'tensor']
        if mode not in modes:
            raise ClassComputationError('mode should be one of {}'.format(mode))
        if mode == 'scalar' and self.pt.has_scalars:
            titles = self.pt.scalar_titles
            data = self.pt.scalar_perturbations_data
            sizes = self.pt.size_scalar_perturbation_data
        elif mode == 'vector' and self.pt.has_vectors:
            titles = self.pt.vector_titles
            data = self.pt.vector_perturbations_data
            sizes = self.pt.size_vector_perturbation_data
        elif mode == 'tensor' and self.pt.has_tensors:
            titles = self.pt.tensor_titles
            data = self.pt.tensor_perturbations_data
            sizes = self.pt.size_tensor_perturbation_data
        else:
            raise ClassRuntimeError('mode {} has not been calculated'.format(mode))
        dtype = _titles_to_dtype(titles)
        ntitles = len(dtype.fields)
        toret = []
        if ntitles:
            for ik in range(self.pt.k_output_values_num):
                timesteps = sizes[ik] // ntitles
                datak = <double[:timesteps, :ntitles]> data[ik]
                array = np.empty(timesteps, dtype=dtype)
                for ititle, title in enumerate(dtype.names):
                    array[title] = datak[:, ititle]
                toret.append(array)
        return toret

    @staticmethod
    cdef get_ic_keys(perturbations *pt, int index_md):
        r"""Helper routine that returns initial condition names."""
        cdef FileName ic_suffix
        cdef char ic_info[_LINE_LENGTH_MAX_]

        cdef ic_num = pt.ic_size[index_md]
        ic_keys = []
        for index_ic in range(ic_num):
            if perturbations_output_firstline_and_ic_suffix(pt, index_ic, ic_info, ic_suffix) == _FAILURE_:
                raise ClassRuntimeError(pt.error_message.decode())
            ic_key = <bytes> ic_suffix
            ic_keys.append(ic_key.decode())
        return ic_keys

    @staticmethod
    cdef get_index_md(perturbations * pt, mode):
        r"""Helper routine that returns indices of 'scalar', 'vector' or 'tensor' perturbations."""
        has_flags = {'scalar': (pt.has_scalars, pt.index_md_scalars == _TRUE_),
                     'vector': (pt.has_vectors, pt.index_md_vectors == _TRUE_),
                     'tensor': (pt.has_tensors, pt.index_md_tensors == _TRUE_)}
        modes = list(has_flags.keys())

        if mode not in modes:
            raise ClassComputationError('mode must be one of {}'.format(modes))
        return has_flags[mode]


cdef class Transfer:

    r"""Wrapper of the ``Transfer` module in CLASS."""

    cdef ClassEngine engine
    cdef perturbations * pt
    cdef background * ba

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Transfer`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('transfer')
        self.pt = &self.engine.pt
        self.ba = &self.engine.ba

    def table(self, mode='scalar'):
        """
        Return the source functions for all (k, z).

        Returns
        -------
        tk : numpy.ndarray
            Structured array of shape (k.size, z.size), or a dictionary of it for each initial condition pair.
        """
        cdef int index_k, index_tau, index_tp
        cdef int index_md
        cdef int tau_size = self.pt.tau_size
        cdef double *** sources = self.pt.sources
        has_mode, index_md = Perturbations.get_index_md(self.pt, mode)
        if not has_mode:
            raise ClassRuntimeError('mode {} has not been calculated'.format(mode))
        cdef int k_size = self.pt.k_size[index_md]
        cdef int tp_size = self.pt.tp_size[index_md]
        ic_keys = Perturbations.get_ic_keys(self.pt, index_md)
        cdef double * k = self.pt.k[index_md];
        cdef double * tau = self.pt.tau_sampling;

        names, indices = [], []

        if self.pt.has_source_t:
            indices.extend([self.pt.index_tp_t0, self.pt.index_tp_t1, self.pt.index_tp_t2])
            names.extend(["t0", "t1", "t2"])
        if self.pt.has_source_p:
            indices.append(self.pt.index_tp_p)
            names.append("p")
        if self.pt.has_source_phi:
            indices.append(self.pt.index_tp_phi)
            names.append("phi")
        if self.pt.has_source_phi_plus_psi:
            indices.append(self.pt.index_tp_phi_plus_psi)
            names.append("phi_plus_psi")
        if self.pt.has_source_phi_prime:
            indices.append(self.pt.index_tp_phi_prime)
            names.append("phi_prime")
        if self.pt.has_source_psi:
            indices.append(self.pt.index_tp_psi)
            names.append("psi")
        if self.pt.has_source_H_T_Nb_prime:
            indices.append(self.pt.index_tp_H_T_Nb_prime)
            names.append("H_T_Nb_prime")
        if self.pt.index_tp_k2gamma_Nb:
            indices.append(self.pt.index_tp_k2gamma_Nb)
            names.append("k2gamma_Nb")
        if self.pt.has_source_h:
            indices.append(self.pt.index_tp_h)
            names.append("h")
        if self.pt.has_source_h_prime:
            indices.append(self.pt.index_tp_h_prime)
            names.append("h_prime")
        if self.pt.has_source_eta:
            indices.append(self.pt.index_tp_eta)
            names.append("eta")
        if self.pt.has_source_eta_prime:
            indices.append(self.pt.index_tp_eta_prime)
            names.append("eta_prime")
        if self.pt.has_source_delta_tot:
            indices.append(self.pt.index_tp_delta_tot)
            names.append("delta_tot")
        if self.pt.has_source_delta_m:
            indices.append(self.pt.index_tp_delta_m)
            names.append("delta_m")
        if self.pt.has_source_delta_cb:
            indices.append(self.pt.index_tp_delta_cb)
            names.append("delta_cb")
        if self.pt.has_source_delta_g:
            indices.append(self.pt.index_tp_delta_g)
            names.append("delta_g")
        if self.pt.has_source_delta_b:
            indices.append(self.pt.index_tp_delta_b)
            names.append("delta_b")
        if self.pt.has_source_delta_cdm:
            indices.append(self.pt.index_tp_delta_cdm)
            names.append("delta_cdm")
        if self.pt.has_source_delta_idm:
            indices.append(self.pt.index_tp_delta_idm)
            names.append("delta_idm")
        if self.pt.has_source_delta_dcdm:
            indices.append(self.pt.index_tp_delta_dcdm)
            names.append("delta_dcdm")
        if self.pt.has_source_delta_fld:
            indices.append(self.pt.index_tp_delta_fld)
            names.append("delta_fld")
        if self.pt.has_source_delta_scf:
            indices.append(self.pt.index_tp_delta_scf)
            names.append("delta_scf")
        if self.pt.has_source_delta_dr:
            indices.append(self.pt.index_tp_delta_dr)
            names.append("delta_dr")
        if self.pt.has_source_delta_ur:
            indices.append(self.pt.index_tp_delta_ur)
            names.append("delta_ur")
        if self.pt.has_source_delta_idr:
            indices.append(self.pt.index_tp_delta_idr)
            names.append("delta_idr")
        if self.pt.has_source_delta_ncdm:
            for incdm in range(self.ba.N_ncdm):
              indices.append(self.pt.index_tp_delta_ncdm1+incdm)
              names.append("delta_ncdm[{}]".format(incdm))
        if self.pt.has_source_theta_tot:
            indices.append(self.pt.index_tp_theta_tot)
            names.append("theta_tot")
        if self.pt.has_source_theta_m:
            indices.append(self.pt.index_tp_theta_m)
            names.append("theta_m")
        if self.pt.has_source_theta_cb:
            indices.append(self.pt.index_tp_theta_cb)
            names.append("theta_cb")
        if self.pt.has_source_theta_g:
            indices.append(self.pt.index_tp_theta_g)
            names.append("theta_g")
        if self.pt.has_source_theta_b:
            indices.append(self.pt.index_tp_theta_b)
            names.append("theta_b")
        if self.pt.has_source_theta_cdm:
            indices.append(self.pt.index_tp_theta_cdm)
            names.append("theta_cdm")
        if self.pt.has_source_theta_idm:
            indices.append(self.pt.index_tp_theta_idm)
            names.append("theta_idm")
        if self.pt.has_source_theta_dcdm:
            indices.append(self.pt.index_tp_theta_dcdm)
            names.append("theta_dcdm")
        if self.pt.has_source_theta_fld:
            indices.append(self.pt.index_tp_theta_fld)
            names.append("theta_fld")
        if self.pt.has_source_theta_scf:
            indices.append(self.pt.index_tp_theta_scf)
            names.append("theta_scf")
        if self.pt.has_source_theta_dr:
            indices.append(self.pt.index_tp_theta_dr)
            names.append("theta_dr")
        if self.pt.has_source_theta_ur:
            indices.append(self.pt.index_tp_theta_ur)
            names.append("theta_ur")
        if self.pt.has_source_theta_idr:
            indices.append(self.pt.index_tp_theta_idr)
            names.append("theta_idr")
        if self.pt.has_source_theta_ncdm:
            for incdm in range(self.ba.N_ncdm):
              indices.append(self.pt.index_tp_theta_ncdm1+incdm)
              names.append("theta_ncdm[{}]".format(incdm))

        cdef double[:] z = np.empty(tau_size, dtype=np.float64)
        for index_tau in range(tau_size):
            if background_z_of_tau(self.ba, tau[index_tau], &(z[index_tau])) == _FAILURE_:
                raise ClassRuntimeError(self.ba.error_message.decode())

        toret = {}
        for index_ic, ic_key in enumerate(ic_keys):
            array = np.empty((k_size, tau_size), dtype=[(name, np.float64) for name in ['k', 'z', 'tau']] + [(str(name), np.float64) for name in names])
            for index_tp, name in zip(indices, names):
                array[name] = <double[:k_size, :tau_size]> sources[index_md][index_ic * tp_size + index_tp]
            array['k'] = <double[:k_size, :1]> k
            array['k'] /= self.ba.h
            array['z'] = z
            array['tau'] = <double[:tau_size]> tau
            toret[ic_key] = array

        if len(ic_keys) == 1:
            return array

        return toret


cdef class Harmonic:

    r"""Wrapper of the ``Harmonic`` module in CLASS."""

    cdef ClassEngine engine
    cdef harmonic * hr
    cdef lensing * le
    cdef precision * pr
    cdef perturbations * pt

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Harmonic`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('harmonic')
        self.hr = &self.engine.hr
        self.le = &self.engine.le
        self.pr = &self.engine.pr
        self.pt = &self.engine.pt

    def unlensed_table(self, ellmax=-1, of=None):
        r"""
        Return table of unlensed :math:`C_{\ell}` (i.e. CMB power spectra without lensing and lensing potentials), unitless.

        Parameters
        ----------
        ellmax : int, default=-1
            Maximum :math:`\ell` desired. If negative, is relative to the requested maximum `\ell`.

        of : list, default=None
            List of outputs, ['tt', 'ee', 'bb', 'te', 'pp', 'tp', 'ep']. If ``None``, return all computed outputs.

        Returns
        -------
        cell : numpy.ndarray
            Structured array.

        Note
        ----
        Normalisation is :math:`C_{\ell}` rather than :math:`\ell(\ell+1)/(2\pi) C_{\ell}` (or :math:`\ell^{2}(\ell+1)^{2}/(2\pi) C_{\ell}` in the case of
        the lensing potential ``pp`` spectrum).
        Usually multiplied by CMB temperature in :math:`\mu K`.
        """
        cdef double *rcl = <double*> malloc(self.hr.ct_size * sizeof(double))

        # Quantities for tensor modes
        cdef double **cl_md = <double**> malloc(self.hr.md_size * sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md[index_md] = <double*> malloc(self.hr.ct_size * sizeof(double))

        # Quantities for isocurvature modes
        cdef double **cl_md_ic = <double**> malloc(self.hr.md_size * sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md_ic[index_md] = <double*> malloc(self.hr.ct_size * self.hr.ic_ic_size[index_md] * sizeof(double))

        #if of is not None:
        #    if any('p' in p for p in of):
        #        self.engine.compute('lensing')

        has_flags = [(self.hr.has_tt, self.hr.index_ct_tt, 'tt'),
                     (self.hr.has_ee, self.hr.index_ct_ee, 'ee'),
                     (self.hr.has_bb, self.hr.index_ct_bb, 'bb'),
                     (self.hr.has_te, self.hr.index_ct_te, 'te'),
                     (self.hr.has_pp, self.hr.index_ct_pp, 'pp'),
                     (self.hr.has_tp, self.hr.index_ct_tp, 'tp'),
                     (self.hr.has_ep, self.hr.index_ct_ep, 'ep')]
        indices = {}
        for flag, index, name in has_flags:
            if of is None:
                if flag == _FALSE_: continue
            elif name not in of: continue
            elif name in of and flag == _FALSE_:
                raise ClassComputationError('You asked for {}, but it has not been calculated.'.format(name))
            indices[name] = index

        names = list(indices.keys())
        dtype = np.dtype([('ell', np.int64)] + [(str(name), np.float64) for name in names])
        if ellmax < 0:
            ellmax = self.hr.l_max_tot + 1 + ellmax
            if self.le.has_lensed_cls:
                ellmax -= self.pr.delta_l_max
        if ellmax > self.hr.l_max_tot:
            raise ClassRuntimeError('You asked for ellmax = {:d}, greater than calculated ellmax = {:d}'.format(ellmax, self.hr.l_max_tot))
        toret = np.empty(ellmax + 1, dtype=dtype)
        toret[:2] = 0
        # Recover for each ell the information from CLASS
        for ell from 2 <= ell < ellmax + 1:
            if harmonic_cl_at_l(self.hr, ell, rcl, cl_md, cl_md_ic) == _FAILURE_:
                raise ClassRuntimeError(self.hr.error_message.decode())
            for name in names:
                toret[name][ell] = rcl[indices[name]]

        toret['ell'] = np.arange(ellmax + 1)
        free(rcl)
        for index_md in range(self.hr.md_size):
            free(cl_md[index_md])
            free(cl_md_ic[index_md])
        free(cl_md)
        free(cl_md_ic)
        return toret

    def lensed_table(self, ellmax=-1, of=None):
        r"""
        Return table of lensed :math:`C_{\ell}`, unitless.

        Parameters
        ----------
        ellmax : int, default=-1
            Maximum :math:`\ell` desired. If negative, is relative to the requested maximum `\ell`.

        of : list, default=None
            List of outputs, ['tt', 'ee', 'bb', 'pp', 'te', 'tp']. If ``None``, return all computed outputs.

        Returns
        -------
        cell : numpy.ndarray
            Structured array.
        """
        self.engine.compute('lensing')

        cdef double *lcl = <double*> malloc(self.le.lt_size * sizeof(double))

        has_flags = [(self.le.has_tt, self.le.index_lt_tt, 'tt'),
                     (self.le.has_ee, self.le.index_lt_ee, 'ee'),
                     (self.le.has_bb, self.le.index_lt_bb, 'bb'),
                     (self.le.has_pp, self.le.index_lt_pp, 'pp'),
                     (self.le.has_te, self.le.index_lt_te, 'te'),
                     (self.le.has_tp, self.le.index_lt_tp, 'tp')]
        indices = {}
        for flag, index, name in has_flags:
            if of is None:
                if flag == _FALSE_: continue
            elif name not in of: continue
            elif name in of and flag == _FALSE_:
                raise ClassComputationError('You asked for lensed {}, but it has not been calculated. Please set lensing = yes.'.format(name))
            indices[name] = index

        names = list(indices.keys())
        dtype = np.dtype([('ell', np.int64)] + [(str(name), np.float64) for name in names])
        if ellmax < 0:
            ellmax = self.le.l_lensed_max + 1 + ellmax
        if ellmax > self.le.l_lensed_max:
            raise ClassRuntimeError('You asked for ellmax = {:d}, greater than calculated ellmax = {:d}'.format(ellmax, self.le.l_lensed_max))
        toret = np.empty(ellmax + 1, dtype=dtype)
        toret[:2] = 0
        # Recover for each ell the information from CLASS

        for ell from 2 <= ell < ellmax + 1:
            if lensing_cl_at_l(self.le, ell, lcl) == _FAILURE_:
                raise ClassRuntimeError(self.le.error_message.decode())
            for name in names:
                toret[name][ell] = lcl[indices[name]]
        toret['ell'] = np.arange(ellmax + 1)

        free(lcl)
        return toret

    def unlensed_cl(self, ellmax=-1):
        r"""Return unlensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        return self.unlensed_table(ellmax=ellmax, of=['tt', 'ee', 'bb', 'te'])

    def lens_potential_cl(self, ellmax=-1):
        r"""Return potential :math:`C_{\ell}` ['pp', 'tp', 'ep'], unitless."""
        return self.unlensed_table(ellmax=ellmax, of=['pp', 'tp', 'ep'])

    def lensed_cl(self, ellmax=-1):
        r"""Return lensed :math:`C_{\ell}` ['tt', 'ee', 'bb', 'te'], unitless."""
        return self.lensed_table(ellmax=ellmax, of=['tt', 'ee', 'bb', 'te'])


cdef class Fourier:

    r"""Wrapper of the ``Fourier` module in CLASS."""

    cdef ClassEngine engine
    cdef precision * pr
    cdef background * ba
    cdef primordial * pm
    cdef perturbations * pt
    cdef fourier * fo
    cdef readonly dict data

    def __init__(self, ClassEngine engine):
        r"""
        Initialise :class:`Fourier`.

        Parameters
        ----------
        engine : ClassEngine
            CLASS engine instance.
        """
        self.engine = engine
        self.engine.compute('fourier')
        self.pr = &self.engine.pr
        self.ba = &self.engine.ba
        self.pm = &self.engine.pm
        self.pt = &self.engine.pt
        self.fo = &self.engine.fo

    property has_non_linear:
        r"""Whether the non-linear power spectrum has been computed."""
        def __get__(self):
          return self.fo.method != nl_none

    property has_pk_m:
        r"""Whether the matter power spectrum has been computed."""
        def __get__(self):
          return self.fo.has_pk_m == _TRUE_

    property has_pk_cb:
        r"""Whether the cold dark matter + baryons power spectrum has been computed."""
        def __get__(self):
          return self.fo.has_pk_cb == _TRUE_

    property sigma8_m:
        r"""Current r.m.s. of matter perturbations in a sphere of :math:`8 \mathrm{Mpc}/h`, unitless."""
        def __get__(self):
            return self.fo.sigma8[self.fo.index_pk_m]

    property sigma8_cb:
        r"""Current r.m.s. of cold dark matter + baryons perturbations in a sphere of :math:`8 \mathrm{Mpc}/h` unitless."""
        def __get__(self):
            return self.fo.sigma8[self.fo.index_pk_cb]

    @gridarray(iargs=[0, 1])
    @cython.boundscheck(False)
    def sigma_rz(self, double[:] r, double[:] z, of='delta_m'):
        r"""
        Return :math:`\sigma_{r}(z)`, r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`.

        Parameters
        ----------
        r : array_like
            Radii in :math:`\mathrm{Mpc}/h`.

        z : array_like
            Redshifts.

        of : string, default='delta_m'
            Perturbed quantities.
            Either 'delta_m' for matter perturbations or 'delta_cb' for cold dark matter + baryons perturbations.

        Returns
        -------
        sigmarz : array_like
            Array of shape ``(r.size, z.size)`` (null dimensions are squeezed).
        """
        cdef int r_size = r.size, z_size = z.size
        cdef double[:,:] toret = np.empty((r_size, z_size), dtype=np.float64)
        cdef int index = self._index_pk_of(of)
        cdef int ir, iz
        with nogil:
            for ir in range(r_size):
                for iz in range(z_size):
                    if fourier_sigmas_at_z(self.pr, self.ba, self.fo, r[ir] / self.ba.h, z[iz], index, out_sigma, &(toret[ir, iz])) == _FAILURE_:
                        toret[ir, iz] = NAN
        return np.asarray(toret)

    @flatarray()
    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    def _index_pk_of(self, of='delta_m'):
        r"""Helper routine that returns index of `of` power spectrum."""
        of = self._check_pk_of(of)
        return {'delta_m': self.fo.index_pk_m, 'delta_cb': self.fo.index_pk_cb}[of]

    def _check_pk_of(self, of='delta_m', silent=True):
        r"""Helper routine that checks requested perturbed quantity `of` has been calculated."""
        if of == 'delta_cb':
            if not self.has_pk_cb:
                if not silent:
                    raise ClassRuntimeError('No cb power spectrum computed.')
                return 'delta_m'
        return of

    def _use_pk_non_linear(self, non_linear=False):
        r"""Helper routine that returns linear or non-linear power spectrum flag."""
        if (self.fo.has_pk_m == _FALSE_):
            raise ClassRuntimeError('No power spectrum computed. You must add mPk to the list of outputs.')
        if non_linear:
            if self.has_non_linear:
                return pk_nonlinear
            raise ClassRuntimeError('You ask CLASS to return an array of non-linear P(k, z) values, '
                                    'but the input parameters sent to CLASS did not require any non-linear P(k, z) calculations; '
                                    'add e.g. "halofit" or "HMcode" in "non_linear"')
        return pk_linear

    @gridarray(iargs=[0, 1])
    def pk_kz(self, np.ndarray k, np.ndarray z, non_linear=False, of='delta_m'):
        r"""
        Return power spectrum, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        z : array_like
            Redshifts.

        non_linear : bool, default=False
            Whether to return the non-linear power spectrum (if requested in parameters, with 'non_linear':'halofit' or 'HMcode').

        of : string, default='delta_m'
            Perturbed quantities.
            Either 'delta_m' for matter perturbations or 'delta_cb' for cold dark matter + baryons perturbations.

        Returns
        -------
        pk : numpy.ndarray
            Power spectrum array of shape (len(k), len(z)).
        """
        cdef pk_outputs is_non_linear = self._use_pk_non_linear(non_linear)
        # internally class uses 1 / Mpc

        k_size, z_size = k.size, z.size
        cdef np.ndarray kh = k * self.ba.h
        cdef np.ndarray pk = np.empty(k_size * z_size, dtype=np.float64)
        cdef np.ndarray pk_cb = np.empty(k_size * z_size, dtype=np.float64)

        fourier_pks_at_kvec_and_zvec(self.ba, self.fo, is_non_linear, <double*> kh.data, k_size, <double*> z.data, z_size, <double*> pk.data, <double*> pk_cb.data)

        toret = pk_cb if self._check_pk_of(of) == 'delta_cb' else pk
        toret[...] *= self.ba.h**3
        return toret

    @cython.boundscheck(False)
    def table(self, non_linear=False, of='delta_m'):
        r"""
        Return power spectrum table, in :math:`(\mathrm{Mpc}/h)^{3}`.

        Parameters
        ----------
        non_linear : bool, default=False
            Whether to return the non-linear power spectrum (if requested in parameters, with 'non_linear':'halofit' or 'HMcode').
            Computed only for ``of == 'delta_m'`` or 'delta_cb'.

        of : string, tuple, default='delta_m'
            Perturbed quantities.
            'delta_m' for matter perturbations, 'delta_cb' for cold dark matter + baryons, 'phi', 'psi' for Bardeen potentials, or 'phi_plus_psi' for Weyl potential.
            Provide a tuple, e.g. ('delta_m', 'theta_cb') for the cross matter density - cold dark matter + baryons velocity power spectra.

        Returns
        -------
        k : numpy.ndarray
            Wavenumbers.

        z : numpy.ndarray
            Redshifts.

        pk : numpy.ndarray
            Power spectrum array of shape (len(k), len(z)).
        """
        cdef pk_outputs is_non_linear = self._use_pk_non_linear(non_linear)
        cdef int index_ln_tau_min
        if is_non_linear == pk_nonlinear:
            index_ln_tau_min = self.fo.index_tau_min_nl - (self.fo.tau_size - self.fo.ln_tau_size)
        else:
            index_ln_tau_min = 0
        cdef int index_ln_tau_max = self.fo.ln_tau_size
        cdef int ln_tau_size = index_ln_tau_max - index_ln_tau_min

        if self.fo.ln_tau_size <= 1:
                raise ClassRuntimeError('You ask CLASS to return an array of P(k, z) values, but the input parameters sent to CLASS did not require '
                                        'any P(k, z) calculations for z>0; pass either a list of z in "z_pk" or one non-zero value in "z_max_pk"')
        if ln_tau_size < 1:  # non_linear
            raise ClassRuntimeError('"halofit_nonlinear_min_k_max" or "hmcode_nonlinear_min_k_max" is too small to compute the scale k_NL for non-linear corrections at any of the requested z_pk.')

        k = np.array(<double[:self.fo.k_size]> self.fo.k, dtype=np.float64)
        cdef double[:] z = np.empty(ln_tau_size, dtype=np.float64)
        cdef double[:,:] pk_at_k_z = np.empty((self.fo.k_size, ln_tau_size), dtype=np.float64)

        cdef int index_k, index_tau, index_tau_late, index_tau_sources
        cdef double z_max_non_linear, z_max_requested

        cdef int index_ic1, index_ic2, index_ic1_ic1, index_ic1_ic2, index_ic2_ic2, index_tp1, index_tp2, ntheta, last_index #junk
        cdef double *primordial_pk = <double*> malloc(self.fo.ic_ic_size * sizeof(double))
        cdef double [:] pvecback
        cdef double ** sources = self.pt.sources[self.fo.index_md_scalars]
        cdef double source_tp1_ic1=0, source_tp2_ic1=0, source_tp1_ic2=0, source_tp2_ic2=0, primordial_pk_ic1_ic2
        cdef double sumpk, factor_z, factor_k

        cdef int pt_tau_size = self.pt.tau_size
        cdef int pt_ln_tau_size = self.pt.ln_tau_size

        for index_tau in range(ln_tau_size):
            index_tau_late = index_tau + index_ln_tau_min
            if index_tau_late == self.fo.ln_tau_size - 1:
                z[index_tau] = 0.
            elif background_z_of_tau(self.ba, exp(self.fo.ln_tau[index_tau_late]), &(z[index_tau])) == _FAILURE_:
                raise ClassRuntimeError(self.ba.error_message.decode())

        if isinstance(of, str): of = (of,)
        of = list(of)
        of = of + [of[0]] * (2 - len(of))

        # Use precomputed spectra
        if of[0] == of[1] and of[0] in ['delta_m', 'delta_cb']:
            index_tp1 = self._index_pk_of(of[0])
            with nogil:
                for index_tau in range(ln_tau_size):
                    index_tau_late = index_tau + index_ln_tau_min
                    for index_k in range(self.fo.k_size):
                        if is_non_linear == pk_nonlinear:
                            pk_at_k_z[index_k, index_tau] = exp(self.fo.ln_pk_nl[index_tp1][index_tau_late * self.fo.k_size + index_k])
                        else:
                            pk_at_k_z[index_k, index_tau] = exp(self.fo.ln_pk_l[index_tp1][index_tau_late * self.fo.k_size + index_k])

        elif is_non_linear == pk_nonlinear:
            raise ClassComputationError('Non-linear power spectrum is computed for auto delta_m and delta_cb only')

        else:
            primordial_pk = <double*> malloc(self.fo.ic_ic_size*sizeof(double))
            pvecback = np.empty(self.ba.bg_size, dtype=np.float64)
            indices = {'delta_m': self.pt.index_tp_delta_m,
                       'delta_cb': self.pt.index_tp_delta_cb if self.pt.has_source_delta_cb == _TRUE_ else self.pt.index_tp_delta_m,
                       'theta_m': self.pt.index_tp_theta_m,
                       'theta_cb': self.pt.index_tp_theta_cb if self.pt.has_source_theta_cb == _TRUE_ else self.pt.index_tp_theta_m,
                       'phi': self.pt.index_tp_phi,
                       'psi': self.pt.index_tp_psi,
                       'phi_plus_psi': self.pt.index_tp_phi_plus_psi}
            index_tp1, index_tp2 = indices[of[0]], indices[of[1]]
            ntheta = sum(of_.startswith('theta_') for of_ in of)

            with nogil:
                for index_tau in range(ln_tau_size):
                    index_tau_late = index_tau + index_ln_tau_min
                    index_tau_sources = pt_tau_size - pt_ln_tau_size + index_tau_late
                    if ntheta > 0:
                        if background_at_z(self.ba, z[index_tau], long_info, inter_normal, &last_index, &pvecback[0]) == _FAILURE_:
                            raise ClassRuntimeError(self.ba.error_message.decode())
                        factor_z = 1. / (-pvecback[self.ba.index_bg_H] * pvecback[self.ba.index_bg_a])**ntheta
                    else:
                        factor_z = 1.
                    for index_k in range(self.fo.k_size):
                        factor_k = 2. * _PI_ * _PI_ / exp(3. * self.fo.ln_k[index_k])
                        if primordial_spectrum_at_k(self.pm, self.fo.index_md_scalars, logarithmic, self.fo.ln_k[index_k], primordial_pk) == _FAILURE_:
                            raise ClassRuntimeError(self.pm.error_message.decode())
                        sumpk = 0.
                        for index_ic1 in range(self.fo.ic_size):
                            index_ic1_ic1 = index_symmetric_matrix(index_ic1, index_ic1, self.fo.ic_size)
                            #source_tp1_ic1 = sources[index_ic1 * tp_size + index_tp1][index_tau_sources * k_size + index_k]

                            if fourier_get_source(self.ba, self.pt, self.fo, index_k, index_ic1, index_tp1, index_tau_sources, sources, &source_tp1_ic1) == _FAILURE_:
                                raise ClassRuntimeError(self.fo.error_message.decode())
                            if index_tp2 != index_tp1:
                                if fourier_get_source(self.ba, self.pt, self.fo, index_k, index_ic1, index_tp2, index_tau_sources, sources, &source_tp2_ic1) == _FAILURE_:
                                    raise ClassRuntimeError(self.fo.error_message.decode())
                            else:
                                source_tp2_ic1 = source_tp1_ic1

                            sumpk += factor_k * source_tp1_ic1 * source_tp2_ic1 * exp(primordial_pk[index_ic1_ic1])
                            for index_ic2 in range(index_ic1+1, self.fo.ic_size):
                                index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic2, self.fo.ic_size)
                                index_ic2_ic2 = index_symmetric_matrix(index_ic2, index_ic2, self.fo.ic_size)
                                if self.fo.is_non_zero[index_ic1_ic2] == _TRUE_:
                                    if fourier_get_source(self.ba, self.pt, self.fo, index_k, index_ic2, index_tp1, index_tau_sources, sources, &source_tp1_ic2) == _FAILURE_:
                                        raise ClassRuntimeError(self.fo.error_message.decode())
                                    if index_tp2 != index_tp1:
                                        if fourier_get_source(self.ba, self.pt, self.fo, index_k, index_ic2, index_tp2, index_tau_sources, sources, &source_tp2_ic2) == _FAILURE_:
                                            raise ClassRuntimeError(self.fo.error_message.decode())
                                        else:
                                            source_tp2_ic2 = source_tp1_ic2
                                    primordial_pk_ic1_ic2 = primordial_pk[index_ic1_ic2] * sqrt(primordial_pk[index_ic1_ic1] * primordial_pk[index_ic2_ic2])
                                    sumpk += factor_k * (source_tp1_ic1 * source_tp2_ic2 + source_tp1_ic2 * source_tp2_ic1) * primordial_pk_ic1_ic2
                        pk_at_k_z[index_k, index_tau] = factor_z * sumpk
            free(primordial_pk)

        return np.asarray(k) / self.ba.h, np.asarray(z), np.asarray(pk_at_k_z) * self.ba.h**3
