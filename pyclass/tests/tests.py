import time

import numpy as np

from pyclass import *


def test_params():
    cosmo = ClassEngine({'N_ncdm': 1, 'm_ncdm': [0.06]})
    d = cosmo.get_params()
    for key, value in d.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_task_dependency():
    #params = {'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'modes': 's,t'}
    params = {'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2'}
    cosmo = ClassEngine({'z_max_pk': 10.0})
    Fourier(cosmo)
    Harmonic(cosmo)

    cosmo = ClassEngine(params)
    Background(cosmo).table()
    Thermodynamics(cosmo).table()
    Primordial(cosmo).table()
    Perturbations(cosmo).table()
    Transfer(cosmo).table()
    Harmonic(cosmo).unlensed_table()
    Fourier(cosmo).table()

    cosmo = ClassEngine(params)
    Fourier(cosmo).table()
    Harmonic(cosmo).lensed_table()
    Transfer(cosmo).table()
    Perturbations(cosmo).table()
    Primordial(cosmo).table()
    Thermodynamics(cosmo).table()
    Background(cosmo).table()

    cosmo = ClassEngine(params)
    Background(cosmo).table()
    cosmo = ClassEngine(params)
    Thermodynamics(cosmo).table()
    cosmo = ClassEngine(params)
    Primordial(cosmo).table()
    cosmo = ClassEngine(params)
    Perturbations(cosmo).table()
    cosmo = ClassEngine(params)
    Transfer(cosmo).table()
    cosmo = ClassEngine(params)
    Harmonic(cosmo).lensed_table()
    cosmo = ClassEngine(params)
    Fourier(cosmo).table()

    #for i in range(10):
    #    cosmo = ClassEngine({'k_output_values': '0.01, 0.2'})
    #    cosmo = ClassEngine(params)
    #    Perturbations(cosmo).table()


def test_background():
    for N in [0, 2]:
        params = {'N_ncdm': N}
        if N: params['m_ncdm'] = [0.06] * N
        cosmo = ClassEngine(params)
        ba = Background(cosmo)

        for name in ['rho_cdm', 'rho_dcdm', 'rho_ncdm_tot', 'p_ncdm_tot', 'Omega_m', 'Omega_pncdm_tot', 'time', 'conformal_time', 'hubble_function', 'comoving_radial_distance', 'comoving_angular_distance', 'growth_factor', 'growth_rate']:
            func = getattr(ba, name)
            assert func(0.1).shape == ()
            assert func([]).shape == (0,)
            assert func(z=np.array([0.1])).shape == (1,)
            assert func(np.array([0.1], dtype='f4')).dtype.itemsize == func(np.array([0.1], dtype='f8')).dtype.itemsize // 2 == 4
            assert np.issubdtype(func(0).dtype, np.floating) and np.issubdtype(func(np.array(0, dtype='i8')).dtype, np.floating)
            assert func(z=[0.1]).shape == (1,)
            assert func([0.2, 0.3]).shape == (2,)
            assert func([[0.2, 0.3, 0.4]]).shape == (1, 3)
            array = np.array([[0.2, 0.3, 0.4]] * 2)
            assert func(array).shape == array.shape == (2, 3)

        for name in ['T_ncdm', 'rho_ncdm', 'p_ncdm', 'Omega_pncdm']:
            func = getattr(ba, name)
            assert func(0.1).shape == (N,)
            assert func([]).shape == (N, 0)
            assert func(np.array([0.1])).shape == (N, 1)
            assert func(np.array([0.1], dtype='f4')).dtype.itemsize == func(np.array([0.1], dtype='f8')).dtype.itemsize // 2 == 4
            assert func(np.array([[0.1], [0.1]])).shape == (N, 2, 1)
            assert func([0.1]).shape == (N, 1,)
            assert func([0.2, 0.3]).shape == (N, 2,)
            assert func([[0.2, 0.3, 0.4]]).shape == (N, 1, 3)

            if N:
                assert func([[0.2, 0.3, 0.4]], species=0).shape == (1, 3)
                assert func([[0.2, 0.3, 0.4]], species=[0]).shape == (1, 1, 3)
            else:
                try: func([[0.2, 0.3, 0.4]], species=0)
                except IndexError: pass
                else: raise ValueError


def test_thermodynamics():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    # cosmo.compute('thermodynamics')
    th = Thermodynamics(cosmo)
    th.z_drag
    th.rs_drag
    th.tau_reio
    th.z_reio
    th.z_rec
    th.rs_rec
    th.theta_star
    th.YHe
    th.table()


def test_primordial():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    pm = Primordial(cosmo)
    assert pm.pk_k([0., 0.1, 0.2]).shape == (3,)
    t = pm.table()


def test_perturbations():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values':'0.01, 0.2'})
    params = {'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values': '0.01, 0.2', 'modes': 's,t', 'A_s': 2e-9}
    cosmo = ClassEngine(params)
    pt = Perturbations(cosmo)
    t1 = pt.table()
    assert len(t1) == 2
    assert t1[0]['tau [Mpc]'].ndim == 1
    sigma81 = Fourier(cosmo).sigma8_m
    params.update(A_s=3e-9)
    cosmo = ClassEngine(params)
    t2 = Perturbations(cosmo).table()
    sigma82 = Fourier(cosmo).sigma8_m
    for name in t2[-1].dtype.names:
        assert np.allclose(t2[-1][name], t1[-1][name]), name
    assert np.allclose(sigma82 / sigma81, (3. / 2.)**0.5)


def test_transfer():
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'N_ncdm': 1, 'm_ncdm': [0.06]})
    tr = Transfer(cosmo)


def test_harmonic():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk tCl pCl lCl', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'lensing':'yes'})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'lensing': 'yes'})
    hr = Harmonic(cosmo)
    hr.lensed_cl(ellmax=10)
    hr.unlensed_cl(ellmax=10)
    hr.lensed_cl(ellmax=-1).size,hr.unlensed_cl(ellmax=-1).size
    assert hr.lensed_cl(ellmax=-1).size == hr.unlensed_cl(ellmax=-1).size


def test_fourier():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    # fo = Perturbations(cosmo)
    fo = Fourier(cosmo)
    for name in ['pk_kz', 'sigma_rz']:
        func = getattr(fo, name)
        assert func(0.1, 0.1).ndim == 0
        assert func(np.array([]), np.array(0.1)).shape == (0,)
        assert func([], []).shape == func(np.array([]), np.array([])).shape == (0, 0)
        assert func([0.1], 0.1).shape == (1,)
        assert func([0.1], [0.1]).shape == (1, 1)
        assert func(10., z=[0., 1.0]).shape == (2,)
        assert func(np.array(10., dtype='f4'), z=np.array([0., 1.0], dtype='f4')).dtype.itemsize == 4
        assert func([1., 10.], [0., 1.0]).shape == (2, 2)
        assert func([[1., 10.]] * 3, [0., 1.0, 2.0]).shape == (3, 2, 3)
    for name in ['sigma8_z']:
        func = getattr(fo, name)
        assert func(0.).shape == ()
        assert func([]).shape == (0,)
        assert func([0., 1.0]).shape == (2,)
        assert func([[0., 1.0]] * 3).shape == (3, 2)
        assert func(np.array([[0., 1.0]] * 3, dtype='f4')).dtype.itemsize == 4
    k, z, pk = fo.table()
    assert pk.shape == (k.size, z.size)


def test_sigma8():
    # cosmo = ClassEngine({'sigma8': 0.8, 'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'sigma8': 0.8, 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    fo = Fourier(cosmo)
    assert abs(fo.sigma8_m - 0.8) < 1e-4
    assert fo.sigma8_z(0., of='delta_m').ndim == 0


def test_classy():
    from classy import Class
    params = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0}
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute(['thermodynamics'])
    cosmo.struct_cleanup()
    cosmo.empty()


def test_classy():

    from classy import Class

    t0 = time.time()
    cosmo = Class()
    #cosmo.set({'output': 'tCl, pCl, lCl'})
    cosmo.set({'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl', 'number_count_contributions': 'density, rsd, lensing'})
    cosmo.compute(level=['harmonic'])
    print('classy', time.time() - t0)

    t0 = time.time()
    cosmo = ClassEngine()
    Harmonic(cosmo)
    print('pyclass', time.time() - t0)

    from cosmoprimo import Cosmology
    t0 = time.time()
    cosmo = Cosmology(engine='class')
    cosmo.get_harmonic()
    cosmo.get_fourier().pk_interpolator(of=('delta_cb', 'delta_cb'))
    print('cosmoprimo', time.time() - t0)

    t0 = time.time()
    cosmo = Class()
    #cosmo.set({'output': 'tCl, pCl, lCl'})
    cosmo.set({'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl', 'number_count_contributions': 'density, rsd, gr'})
    cosmo.compute(level=['fourier'])
    print('classy', time.time() - t0)

    t0 = time.time()
    cosmo = ClassEngine()
    Fourier(cosmo)
    print('pyclass', time.time() - t0)

    from cosmoprimo import Cosmology
    t0 = time.time()
    cosmo = Cosmology(engine='class')
    cosmo.get_fourier()
    print('cosmoprimo', time.time() - t0)

    #import classy
    #print(classy.__file__)
    """
    for i in range(50):
        cosmo = ClassEngine({'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl', 'modes': 's,t'})
        #Perturbations(cosmo)
        cosmo.compute('perturbations')
    """

    cosmo = Class()
    cosmo.set({'z_pk': '1, 2, 3', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl'})
    cosmo.compute(level=['fourier'])
    cosmo.compute(level=['harmonic'])

    cosmo = ClassEngine({'z_pk': '1, 2, 3'})
    Fourier(cosmo)
    Harmonic(cosmo)

    cosmo = Class()
    cosmo.set({'z_pk': '1, 2, 3', 'output': 'dTk, mPk', 'gauge': 'newton'})
    cosmo.compute()
    pk_ref, k_ref, z_ref = cosmo.get_Weyl_pk_and_k_and_z(nonlinear=False)
    pk_ref_2, k_ref, z_ref = cosmo.get_pk_and_k_and_z(nonlinear=False)
    print(pk_ref[:, -1] / pk_ref_2[:, -1])
    cosmo = ClassEngine({'z_pk': '1, 2, 3', 'gauge': 'newton'})
    ba = Background(cosmo)
    rho_m = 3. / 2. * ba.rho_m(0.) / ba._RH0_
    print(rho_m**2)

    k, z, pk = Fourier(cosmo).table(of='phi_plus_psi')
    pk /= ba.h**3
    pk *= k[:, None]**4
    pk /= 4
    print(pk[:, -3:] / pk_ref[:, -3:])

    for i in range(50):
        cosmo = Class()
        #cosmo.set({'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl', 'modes': 's,t'})
        #cosmo.set({'k_output_values': '0.01, 0.2', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl', 'modes': 's,t'})
        cosmo.set({'k_output_values': '0.01', 'output': 'dTk,vTk,tCl,pCl,mPk', 'modes': 's,t'})
        cosmo.compute(level=['perturb'])
        cosmo.struct_cleanup()
        cosmo.empty()


def test_error():
    params = {'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'Omega_m': 0.1, 'Omega_b': 0.12}
    cosmo = ClassEngine(params)
    Background(cosmo).table()


def test_rs_drag():
    #params = {'A_s': 2.1266796357944756e-09, 'n_s': 0.9657119, 'H0': 70.0, 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'tau_reio': 0.0544, 'N_ncdm': 0, 'output': 'mPk', 'z_pk': 0.61, 'P_k_max_1/Mpc': 1}
    params = {'Omega_k': -0.0006291943, 'k_pivot': 0.05, 'n_s': 0.9649, 'alpha_s': 0.0, 'T_cmb': 2.7255, 'reionization_width': 0.5, 'A_L': 1.0,
              'modes': 's', 'YHe': 'BBN', 'A_s': 2.083e-09, 'Omega_cdm': 0.25975577211637285, 'Omega_b': 0.049431554422906136,
              'tau_reio': 0.05853273, 'h': 0.6766846, 'N_ur': 2.0328, 'm_ncdm': 0.05999991930682943, 'lensing': 'no', 'z_max_pk': 10.0,
              'P_k_max_h/Mpc': 10.0, 'l_max_scalars': 2500, 'N_ncdm': 1, 'T_ncdm': 0.71611, 'recombination': 'HyRec', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl'}
    cosmo = ClassEngine(params)
    ba = Background(cosmo)
    th = Thermodynamics(cosmo)
    print(th.rs_drag / ba.h)

    from classy import Class
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute(level=['perturb'])
    print('classy', cosmo.rs_drag())


    params = {'Omega_k': -0.0006291943, 'k_pivot': 0.05, 'n_s': 0.9649, 'alpha_s': 0.0, 'T_cmb': 2.7255, 'reionization_width': 0.5, 'A_L': 1.0,
              'modes': 's', 'YHe': 'BBN', 'A_s': 2.083e-09, 'Omega_cdm': 0.25975577211637285, 'Omega_b': 0.049431554422906136, 'tau_reio': 0.05853273,
              'h': 0.6766846, 'N_ur': 2.0328, 'm_ncdm': [0.05999991930682943], 'lensing': 'no', 'z_max_pk': 10.0, 'P_k_max_h/Mpc': 10.0,
              'l_max_scalars': 2500, 'N_ncdm': 1, 'T_ncdm': [0.71611], 'recombination': 'HyRec'}
    cosmo = ClassEngine(params)
    ba = Background(cosmo)
    th = Thermodynamics(cosmo)
    print(th.rs_drag / ba.h)

    params['sBBN file'] = 'bbn/sBBN.dat'
    cosmo = ClassEngine(params)
    ba2 = Background(cosmo)
    th2 = Thermodynamics(cosmo)
    assert not np.allclose(th2.rs_drag, th.rs_drag)
    print(th2.rs_drag / ba2.h, th2.rs_drag / th.rs_drag)

    from classy import Class

    cosmo = Class()
    params = {'Omega_k': -0.0006291943, 'k_pivot': 0.05, 'n_s': 0.9649, 'alpha_s': 0.0, 'T_cmb': 2.7255, 'reionization_width': 0.5, 'A_L': 1.0,
              'modes': 's', 'YHe': 'BBN', 'A_s': 2.083e-09, 'Omega_cdm': 0.25975577211637285, 'Omega_b': 0.049431554422906136,
              'tau_reio': 0.05853273, 'h': 0.6766846, 'N_ur': 2.0328, 'm_ncdm': 0.05999991930682943, 'lensing': 'no', 'z_max_pk': 10.0,
              'P_k_max_h/Mpc': 10.0, 'l_max_scalars': 2500, 'N_ncdm': 1, 'T_ncdm': 0.71611, 'recombination': 'HyRec', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl'}
    params = {#'hyrec_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/HyRec2020/',
              #'Galli_file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/heating/Galli_et_al_2013.dat',
              #'sd_external_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/distortions/',
              #'sBBN file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/bbn/sBBN.dat',

              #'hyrec_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/class_public/external/HyRec2020/',
              #'Galli_file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/class_public/external/heating/Galli_et_al_2013.dat',
              #'sd_external_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/class_public/external/distortions/',
              #'sBBN file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/class_public/external/bbn/sBBN.dat',
              'Omega_k': '-0.0006291943', 'k_pivot': '0.05', 'n_s': '0.9649', 'alpha_s': '0.0', 'T_cmb': '2.7255', 'reionization_width': '0.5', 'A_L': '1.0',
              'modes': 's', 'YHe': 'BBN', 'A_s': '2.083e-09', 'Omega_cdm': '0.25975577211637285', 'Omega_b': '0.049431554422906136',
              'tau_reio': '0.05853273', 'h': '0.6766846', 'N_ur': '2.0328', 'm_ncdm': '0.05999991930682943', 'lensing': 'no',
              'z_max_pk': '10.0', 'P_k_max_h/Mpc': '10.0', 'l_max_scalars': '2500', 'N_ncdm': '1', 'T_ncdm': '0.71611',
              'recombination': 'HyRec', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl'}
    cosmo.set(**params)
    cosmo.compute(level=['thermodynamics'])
    print(cosmo.rs_drag())


def test_cl():
    params = {'Omega_k': -0.0006291943, 'k_pivot': 0.05, 'n_s': 0.9649, 'alpha_s': 0.0, 'T_cmb': 2.7255, 'reionization_width': 0.5, 'A_L': 1.0,
              'modes': 's', 'YHe': 'BBN', 'A_s': 2.083e-09, 'Omega_cdm': 0.25975577211637285, 'Omega_b': 0.049431554422906136,
              'tau_reio': 0.05853273, 'h': 0.6766846, 'N_ur': 2.0328, 'm_ncdm': 0.05999991930682943, 'lensing': 'no', 'z_max_pk': 10.0,
              'P_k_max_h/Mpc': 10.0, 'l_max_scalars': 2500, 'N_ncdm': 1, 'T_ncdm': 0.71611,
              'recombination': 'HyRec', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl', 'lensing': 'yes', 'non_linear': 'hmcode'}
    cosmo = ClassEngine(params)
    cl = Harmonic(cosmo).lensed_cl()['tt']
    params = {'Omega_k': '-0.0006291943', 'k_pivot': '0.05', 'n_s': '0.9649', 'alpha_s': '0.0', 'T_cmb': '2.7255', 'reionization_width': '0.5', 'A_L': '1.0',
              'modes': 's', 'YHe': 'BBN', 'A_s': '2.083e-09', 'Omega_cdm': '0.25975577211637285', 'Omega_b': '0.049431554422906136',
              'tau_reio': '0.05853273', 'h': '0.6766846', 'N_ur': '2.0328', 'm_ncdm': '0.05999991930682943', 'lensing': 'yes',
              'z_max_pk': '10.0', 'P_k_max_h/Mpc': '10.0', 'l_max_scalars': '2500', 'N_ncdm': '1', 'T_ncdm': '0.71611',
              'recombination': 'HyRec', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl', 'lensing': 'yes', 'non_linear': 'hmcode'}

    from classy import Class

    cosmo = Class()
    cosmo.set(**params)
    cosmo.compute()
    print(cosmo.lensed_cl()['tt'] / cl)

    params = {'A_s': 2.1266796357944756e-09, 'n_s': 0.9657119, 'H0': 70.0, 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'tau_reio': 0.0544, 'N_ncdm': 0, 'N_ur': 3.044, 'output': 'tCl pCl lCl', 'non_linear': 'hmcode', 'l_max_scalars': 2508, 'lensing': 'yes'}

    #params = {#'hyrec_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/HyRec2020/',
    #          #'Galli_file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/heating/Galli_et_al_2013.dat',
    #          #'sd_external_path': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/distortions/',
    #          #'sBBN file': '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/pyclass/pyclass/base/external/bbn/sBBN_2017.dat',
    #          'A_s': '2.126679635794475617e-09', 'n_s': '9.657118999999999565e-01', 'H0': '7.000000000000000000e+01',
    #          'omega_b': '2.246305999999999997e-02', 'omega_cdm': '1.184774999999999995e-01', 'tau_reio': '5.439999999999999697e-02',
    #          'N_ncdm': '0.000000000000000000e+00', 'N_ur': '3.044000000000000039e+00', 'output': 'tCl lCl pCl', 'non_linear': 'hmcode',
    #          'l_max_scalars': '2.508000000000000000e+03', 'lensing': 'yes', 'reionization_width': '0.5', 'A_L': '1.0', 'modes': 's'}

    cosmo = ClassEngine(params)
    cl = Harmonic(cosmo).lensed_cl()['tt']
    from classy import Class
    cosmo = Class()
    cosmo.set(**params)
    cosmo.compute()
    print(np.abs(cosmo.lensed_cl()['tt'][2:] / cl[2:] - 1.).max())


def test_axiclass(show=False):
    #NEW: addition for EDE (Rafaela Gsponer)
    from pyclass.axiclass import ClassEngine, Background, Fourier
    for fede in [0.001, 0.05, 0.132, 0.15]:
        params = {'omega_b': 0.02251, 'omega_cdm': 0.1320, 'H0': 72.81, 'tau_reio': 0.068, 'scf_potential': 'axion', 'n_axion': 2.6, 'log10_axion_ac': -3.531, 'fraction_axion_ac': fede, 'scf_parameters': [2.72, 0.0], 'scf_evolve_as_fluid': False,
                  'scf_evolve_like_axionCAMB': False, 'attractor_ic_scf': False, 'compute_phase_shift': False, 'include_scf_in_delta_m': True, 'include_scf_in_delta_cb': True}
        cosmo = ClassEngine(params)
        ba = Background(cosmo)
        fo = Fourier(cosmo)
        k = np.logspace(-4, np.log10(3), 1000)
        h = ba.h
        pk = fo.pk_kz(k * h, 0) * h**3
        if show:
            from matplotlib import pyplot as plt
            plt.loglog(k, pk, label=r'$f_\mathrm{{EDE}} = {:.4f}$'.format(fede))
    if show:
        plt.legend()
        plt.xlabel(r'$k$ $[h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)$ $[(\mathrm{Mpc}/h)^3]$')
        plt.show()


def test_edeclass(show=False):
    #NEW: addition for EDE
    from pyclass.edeclass import ClassEngine, Background, Fourier
    for fede in [0.001, 0.05, 0.132, 0.15]:
        '''
        params = {'omega_b': 0.02251, 'omega_cdm': 0.1320, 'H0': 72.81, 'tau_reio': 0.068,
                  'attractor_ic_scf': 'no', 'scf_parameters': [1, 1, 1, 1, 1, 0.0],
                  'log10f_scf': 26., 'm_scf': 1.e-26, 'thetai_scf': 2.6, 'n_scf': 3,
                  'CC_scf': 1, 'scf_tuning_index': 3} #, 'fEDE': fede, 'log10z_c': 3.12}
        '''
        params = {'omega_b': 0.02251, 'omega_cdm': 0.1320, 'H0': 72.81, 'tau_reio': 0.068,
                  'log10f_scf': 26., 'm_scf': 1.e-26, 'thetai_scf': 2.6, 'n_scf': 3, 'CC_scf': 1}
        cosmo = ClassEngine(params)
        ba = Background(cosmo)
        fo = Fourier(cosmo)
        k = np.logspace(-4, np.log10(3), 1000)
        h = ba.h
        pk = fo.pk_kz(k * h, 0) * h**3
        if show:
            from matplotlib import pyplot as plt
            plt.loglog(k, pk, label=r'$f_\mathrm{{EDE}} = {:.4f}$'.format(fede))
    if show:
        plt.legend()
        plt.xlabel(r'$k$ $[h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)$ $[(\mathrm{Mpc}/h)^3]$')
        plt.show()


def test_mochiclass(show=False):
    from pyclass.mochiclass import ClassEngine, Background, Fourier

    params = {'H0': 75.50415, 'omega_cdm': 0.1242302, 'omega_b': 0.02172526, 'tau_reio': 0.05206174, 'A_s': 2.051785e-09, 'n_s': 0.9548171, 'Omega_Lambda': 0, 'Omega_fld': 0,
              'Omega_smg': -1, 'gravity_model': 'brans dicke', 'parameters_smg': [0.7, 50, 1., 1.e-1],
              'skip_stability_tests_smg': 'no', 'a_min_stability_test_smg': 1e-6}
    cosmo = ClassEngine(params)
    ba = Background(cosmo)
    fo = Fourier(cosmo)
    k = np.logspace(-4, np.log10(3), 1000)
    h = ba.h
    pk = fo.pk_kz(k * h, 0) * h**3
    if show:
        from matplotlib import pyplot as plt
        plt.loglog(k, pk)
        plt.xlabel(r'$k$ $[h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)$ $[(\mathrm{Mpc}/h)^3]$')
        plt.show()


def test_negnuclass(show=False):
    #NEW: addition for EDE (Rafaela)
    from pyclass.negnuclass import ClassEngine, Background, Fourier

    params = {'H0': 75.50415, 'omega_cdm': 0.1242302, 'omega_b': 0.02172526, 'tau_reio': 0.05206174, 'A_s': 2.051785e-09, 'n_s': 0.9548171, 'N_ncdm': 1}
    pks = []
    for m_ncdm in [-0.4, 0.001]:
        params.update(m_ncdm=[m_ncdm])
        cosmo = ClassEngine(params)
        ba = Background(cosmo)
        fo = Fourier(cosmo)
        k = np.logspace(-4, np.log10(3), 1000)
        h = ba.h
        pk = fo.pk_kz(k * h, 0, of='theta_cb') * h**3
        pks.append(pk)
    if show:
        from matplotlib import pyplot as plt
        for pk in pks:
            plt.loglog(k, pk)
        plt.xlabel(r'$k$ $[h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)$ $[(\mathrm{Mpc}/h)^3]$')
        plt.show()


if __name__ == '__main__':

    #test_classy()
    test_params()
    test_task_dependency()
    test_background()
    test_thermodynamics()
    test_primordial()
    test_perturbations()
    test_transfer()
    test_harmonic()
    test_fourier()
    test_sigma8()
    test_axiclass(show=True)
    test_mochiclass(show=True)
    test_negnuclass(show=True)
    test_edeclass(show=True)
