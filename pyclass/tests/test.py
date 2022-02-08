import pytest
import numpy as np

from pyclass import *


def test_params():
    cosmo = ClassEngine({'N_ncdm': 1, 'm_ncdm':[0.06]})
    d = cosmo.get_params()
    for key, value in d.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_task_dependency():
    params = {'P_k_max_h/Mpc': 2., 'z_max_pk': 10.0, 'k_output_values':'0.01, 0.2','modes':'s,t'}
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


def test_background():
    cosmo = ClassEngine({'N_ncdm': 1, 'm_ncdm':[0.06]})
    ba = Fourier(cosmo)
    ba = Background(cosmo)
    assert ba.hubble_function(0.1).shape == ()
    assert ba.hubble_function(np.array([0.1])).shape == (1,)
    assert ba.hubble_function([0.1]).shape == (1,)
    assert ba.hubble_function([0.2, 0.3]).shape == (2,)
    assert ba.hubble_function([[0.2, 0.3, 0.4]]).shape == (1, 3)
    ba.time([[0.2, 0.3, 0.4]])
    assert not np.any(np.isnan(ba.comoving_radial_distance([[0.2, 0.3, 0.4]])))
    ba.angular_diameter_distance([[0.2, 0.3, 0.4]])
    ba.luminosity_distance([[0.2, 0.3, 0.4]])
    ba.hubble_function([[0.2, 0.3, 0.4]])
    ba.hubble_function_prime([[0.2, 0.3, 0.4]])
    ba.growth_factor([[0.2, 0.3, 0.4]])
    ba.growth_rate([[0.2, 0.3, 0.4]])
    t = ba.table()


def test_thermodynamics():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    #cosmo.compute('thermodynamics')
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
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    pm = Primordial(cosmo)
    assert pm.pk_k([0., 0.1, 0.2]).shape == (3,)
    t = pm.table()


def test_perturbations():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values':'0.01, 0.2'})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'k_output_values':'0.01, 0.2','modes':'s,t'})
    pt = Perturbations(cosmo)
    assert len(pt.table()) == 2


def test_transfer():
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'N_ncdm': 1, 'm_ncdm':[0.06]})
    tr = Transfer(cosmo)
    t = tr.table(0.0)
    t = tr.table(2.0)
    assert 'd_ncdm[0]' in tr.table().dtype.names


def test_harmonic():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk tCl pCl lCl', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'lensing':'yes'})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0, 'lensing':'yes'})
    hr = Harmonic(cosmo)
    hr.lensed_cl(ellmax=10)
    hr.unlensed_cl(ellmax=10)
    hr.lensed_cl(ellmax=-1).size,hr.unlensed_cl(ellmax=-1).size
    assert hr.lensed_cl(ellmax=-1).size == hr.unlensed_cl(ellmax=-1).size


def test_fourier():
    #cosmo = ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    #fo = Perturbations(cosmo)
    fo = Fourier(cosmo)
    assert np.isscalar(fo.pk_kz(0.1, 0.1))
    assert fo.pk_kz([0.1], 0.1).shape == (1,)
    assert fo.pk_kz([0.1], [0.1]).shape == (1,1)
    assert fo.sigma8_z([0., 1.0]).shape == (2,)
    assert fo.sigma_rz(10., [0., 1.0]).shape == (2,)
    assert fo.sigma_rz([1.,10.], [0., 1.0]).shape == (2,2)
    k, z, pk = fo.table()
    assert pk.shape == (k.size,z.size)


def test_sigma8():
    #cosmo = ClassEngine({'sigma8': 0.8, 'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    cosmo = ClassEngine({'sigma8': 0.8, 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0})
    fo = Fourier(cosmo)
    assert abs(fo.sigma8_m - 0.8) < 1e-4
    assert np.isscalar(fo.sigma8_z(z=0,of='delta_m'))


def test_classy():
    import classy
    from classy import Class
    params = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc': 20., 'z_max_pk': 100.0}
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute(['thermodynamics'])
    cosmo.struct_cleanup()
    cosmo.empty()


if __name__ == '__main__':

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
