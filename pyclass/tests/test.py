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
        for name in ['rho_cdm', 'rho_dcdm', 'Omega_m', 'Omega_pncdm_tot', 'time', 'hubble_function', 'comoving_radial_distance', 'comoving_angular_distance', 'growth_factor', 'growth_rate']:
            func = getattr(ba, name)
            assert func(0.1).shape == ()
            assert func([]).shape == (0,)
            assert func(np.array([0.1])).shape == (1,)
            assert func(np.array([0.1], dtype='f4')).dtype.itemsize == func(np.array([0.1], dtype='f8')).dtype.itemsize // 2 == 4
            assert np.issubdtype(func(0).dtype, np.floating) and np.issubdtype(func(np.array(0, dtype='i8')).dtype, np.floating)
            assert func([0.1]).shape == (1,)
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


def test_thermodynamics():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    # cosmo.compute('thermodynamics')
    th = Thermodynamics(cosmo)
    z_d = th.z_drag
    rs_d = th.rs_drag
    tau_reio = th.tau_reio
    z_reio = th.z_reio
    z_rec = th.z_rec
    rs_res = th.rs_rec
    theta_star = th.theta_star
    t = th.table()


def test_primordial():
    # cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    pm = Primordial(cosmo)
    assert pm.pk_k([0., 0.1, 0.2]).shape == (3,)
    t = pm.table()


def test_perturbations():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values':'0.01, 0.2'})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values': '0.01, 0.2', 'modes': 's,t'})
    pt = Perturbations(cosmo)
    t = pt.table()
    assert len(t) == 2
    assert t[0]['tau [Mpc]'].ndim == 1


def test_transfer():
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'N_ncdm': 1, 'm_ncdm': [0.06]})
    tr = Transfer(cosmo)
    t = tr.table(0.0)
    t = tr.table(2.0)
    assert 'd_ncdm[0]' in tr.table().dtype.names


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
        assert func(10., [0., 1.0]).shape == (2,)
        assert func(np.array(10., dtype='f4'), np.array([0., 1.0], dtype='f4')).dtype.itemsize == 4
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
    #import classy
    #print(classy.__file__)
    """
    for i in range(50):
        cosmo = ClassEngine({'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'output': 'dTk, vTk, tCl, pCl, lCl, mPk, nCl', 'modes': 's,t'})
        #Perturbations(cosmo)
        cosmo.compute('perturbations')
    """
    from classy import Class

    for i in range(50):
        cosmo = Class()
        #cosmo.set({'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values': '0.01, 0.2', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl', 'modes': 's,t'})
        #cosmo.set({'k_output_values': '0.01, 0.2', 'output': 'dTk,vTk,tCl,pCl,lCl,mPk,nCl', 'modes': 's,t'})
        cosmo.set({'k_output_values': '0.01', 'output': 'dTk,vTk,tCl,pCl,mPk', 'modes': 's,t'})
        cosmo.compute(level=['perturb'])
        cosmo.struct_cleanup()
        cosmo.empty()


if __name__ == '__main__':

    test_background()
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
