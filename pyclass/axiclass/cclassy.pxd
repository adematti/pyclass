# Bunch of declarations from C to python. The idea here is to define only the
# quantities that will be used, for input, output or intermediate manipulation,
# by the python wrapper. For instance, in the precision structure, the only
# item used here is its error message. That is why nothing more is defined from
# this structure. The rest is internal in Class.
# If, for whatever reason, you need an other, existing parameter from Class,
# remember to add it inside this cdef.

DEF _MAX_NUMBER_OF_K_FILES_ = 30
DEF _MAXTITLESTRINGLENGTH_ = 8000
DEF _FILENAMESIZE_ = 256
DEF _LINE_LENGTH_MAX_ = 1024
DEF _ARGUMENT_LENGTH_MAX_ = 1024
DEF _ERRORMSGSIZE_ = 2048


cdef extern from "class.h":

    cdef char[10] _VERSION_

    ctypedef char FileArg[_ARGUMENT_LENGTH_MAX_]

    ctypedef char ErrorMsg[_ERRORMSGSIZE_]

    ctypedef char FileName[_FILENAMESIZE_]

    cdef enum interpolation_method:
        inter_normal
        inter_growing_closeby

    cdef enum vecback_format:
        short_info
        normal_info
        long_info

    cdef enum linear_or_logarithmic:
        linear
        logarithmic

    cdef enum possible_gauges:
        newtonian
        synchronous

    cdef enum file_format:
        class_format
        camb_format

    cdef enum non_linear_method:
        nl_none
        nl_halofit
        nl_HMcode

    cdef enum pk_outputs:
        pk_linear
        pk_nonlinear

    cdef enum out_sigmas:
        out_sigma
        out_sigma_prime
        out_sigma_disp

    cdef struct precision:
        ErrorMsg error_message
        int delta_l_max

    cdef struct background:
        ErrorMsg error_message
        short has_cdm
        short has_ncdm
        short has_lambda
        short has_fld
        short has_ur
        short has_dcdm
        int bg_size
        int index_bg_a
        int index_bg_ang_distance
        int index_bg_lum_distance
        int index_bg_conf_distance
        int index_bg_time
        int index_bg_H
        int index_bg_H_prime
        int index_bg_D
        int index_bg_f
        int index_bg_Omega_r
        int index_bg_Omega_m
        int index_bg_rho_g
        int index_bg_rho_b
        int index_bg_rho_cdm
        int index_bg_rho_dcdm
        int index_bg_rho_fld
        int index_bg_rho_lambda
        int index_bg_w_fld
        int index_bg_rho_ur
        int index_bg_rho_crit
        int index_bg_rho_ncdm1
        int index_bg_p_ncdm1

        int sgnK
        double T_cmb
        double * T_ncdm
        double H0
        double h
        double age
        double conformal_age
        int N_ncdm
        double * m_ncdm_in_eV
        double Neff
        double Omega0_b
        double Omega0_g
        double Omega0_cdm
        double Omega0_dcdm
        double Omega0_r
        double Omega0_m
        double Omega0_ur
        double Omega0_ncdm_tot
        double * Omega0_ncdm
        double Omega0_lambda
        double Omega0_fld
        double Omega0_k
        double w0_fld
        double wa_fld
        double cs2_fld
        double K
        int bt_size

        #NEW: addition for EDE (Rafaela) (maybe also crosscheck with input.c)
        #start
        double f_ede
        double m_scf
        double f_axion
        double Omega_axion_ac
        double log10_axion_ac
        double log10_z_c
        double log10_f_axion
        double log10_m_axion
        double omega_axion
        double phi_scf_c
        double phi_ini_scf
        double V0_phi2n
        double f_ede_peak
        double a_peak
        double Omega0_axion
        double Omega0_scf
        double Omega_EDE

        double n_axion_security
        double security_small_Omega_scf
        short scf_evolve_as_fluid
        double threshold_scf_fluid_m_over_H
        short attractor_ic_scf
        int scf_tuning_index
        double n_axion
        short scf_evolve_like_axionCAMB
        short scf_has_perturbations
        short cs2_is_wn

        # wanted to use these three parameters
        # but seems like there are nowhere to be found in the class code (besides declaration)

        # double precision_newton_method_x
        # double precision_newton_method_F
        # double adptative_stepsize

        # additional parameters but doesn't seem necessary
        # to include them atm

#         double phi_ini_scf
#         double phi_prime_ini_scf
#         enum scf_pot scf_potential
#         int scf_parameters_size
#         double beta_scf
#         double alpha_squared
#         double power_of_mu
#         double V0_phi2n
#         double a_c
#         double log10_fraction_axion_ac
#         double Omega_axion_ac
#         double zc_is_zeq
#         double phi_scf_c
#         double w_scf
#         double cs2_scf

#         short scf_kg_eq
#         short kg_fld_switch
#         short scf_fluid_eq
#         short loop_over_background_for_closure_relation
#         short include_scf_in_growth_factor

#         double nu_fld
#         double n_pheno_axion
#         double omega_axion
#         double Theta_initial_fld
#         double m_fld
#         double alpha_fld
#         double Omega_fld_ac
#         double n_cap_infinity
#         double w_fld_f
#         double w_fld_i
#         double log10_a_c
#         double a_peak
#         double cs2_fld

        double * scf_parameters

        #end


    cdef struct thermodynamics:
        ErrorMsg error_message

        int th_size
        int index_th_xe
        int index_th_Tb
        double z_reio
        double tau_reio
        double z_rec
        double tau_rec
        double z_star
        double rs_rec
        double ra_rec
        double ds_rec
        double da_rec
        double z_d
        double tau_d
        double ds_d
        double rs_d
        double rs_star
        double da_star
        double YHe
        double n_e

        int tt_size


    cdef struct perturbations:
        ErrorMsg error_message
        short has_scalars
        short has_vectors
        short has_tensors

        short has_cl_cmb_temperature
        short has_cl_cmb_polarization
        short has_cl_cmb_lensing_potential
        short has_density_transfers
        short has_velocity_transfers
        short has_metricpotential_transfers
        short has_cls
        short has_pk_matter
        short has_pk_cb
        short has_cl_number_count
        short has_nc_density
        short has_nc_rsd
        short has_nc_lens
        short has_nc_gr

        # add source functions for comparison
        short has_source_t
        short has_source_p
        short has_source_delta_m
        short has_source_delta_cb
        short has_source_delta_tot
        short has_source_delta_g
        short has_source_delta_b
        short has_source_delta_cdm
        short has_source_delta_idm
        short has_source_delta_idr
        short has_source_delta_dcdm
        short has_source_delta_fld
        short has_source_delta_scf
        short has_source_delta_dr
        short has_source_delta_ur
        short has_source_delta_ncdm
        short has_source_theta_m
        short has_source_theta_cb
        short has_source_theta_tot
        short has_source_theta_g
        short has_source_theta_b
        short has_source_theta_cdm
        short has_source_theta_idm
        short has_source_theta_idr
        short has_source_theta_dcdm
        short has_source_theta_fld
        short has_source_theta_scf
        short has_source_theta_dr
        short has_source_theta_ur
        short has_source_theta_ncdm
        short has_source_phi
        short has_source_phi_prime
        short has_source_phi_plus_psi
        short has_source_psi
        short has_source_h
        short has_source_h_prime
        short has_source_eta
        short has_source_eta_prime
        short has_source_H_T_Nb_prime
        short has_source_k2gamma_Nb

        int index_tp_t0
        int index_tp_t1
        int index_tp_t2
        int index_tp_p
        int index_tp_delta_m
        int index_tp_delta_cb
        int index_tp_delta_tot
        int index_tp_delta_g
        int index_tp_delta_b
        int index_tp_delta_cdm
        int index_tp_delta_idm
        int index_tp_delta_dcdm
        int index_tp_delta_fld
        int index_tp_delta_scf
        int index_tp_delta_dr
        int index_tp_delta_ur
        int index_tp_delta_idr
        int index_tp_delta_ncdm1
        int index_tp_theta_m
        int index_tp_theta_cb
        int index_tp_theta_tot
        int index_tp_theta_g
        int index_tp_theta_b
        int index_tp_theta_cdm
        int index_tp_theta_dcdm
        int index_tp_theta_fld
        int index_tp_theta_scf
        int index_tp_theta_ur
        int index_tp_theta_idr
        int index_tp_theta_idm
        int index_tp_theta_dr
        int index_tp_theta_ncdm1
        int index_tp_phi
        int index_tp_phi_prime
        int index_tp_phi_plus_psi
        int index_tp_psi
        int index_tp_h
        int index_tp_h_prime
        int index_tp_eta
        int index_tp_eta_prime
        int index_tp_H_T_Nb_prime
        int index_tp_k2gamma_Nb


        double *** sources
        double * tau_sampling
        int tau_size
        int k_size_pk
        int * k_size
        double ** k
        int * ic_size
        int index_ic_ad
        int md_size
        int * tp_size
        double * ln_tau
        int ln_tau_size

        int index_md_scalars
        int index_md_vectors
        int index_md_tensors

        possible_gauges gauge
        double k_min
        double k_max_for_pk
        double z_max_pk
        int l_scalar_max
        int l_lss_max

        #NEW: addition for EDE (Rafaela)
        #start

        double phase_shift
        double amplitude

        short use_big_theta_scf
        short use_big_theta_fld
        short compute_phase_shift
        short include_scf_in_delta_m
        short include_scf_in_delta_cb

        #end

        int store_perturbations
        int k_output_values_num
        double k_output_values[_MAX_NUMBER_OF_K_FILES_]
        int index_k_output_values[_MAX_NUMBER_OF_K_FILES_]
        char scalar_titles[_MAXTITLESTRINGLENGTH_]
        char vector_titles[_MAXTITLESTRINGLENGTH_]
        char tensor_titles[_MAXTITLESTRINGLENGTH_]
        int number_of_scalar_titles
        int number_of_vector_titles
        int number_of_tensor_titles

        double * scalar_perturbations_data[_MAX_NUMBER_OF_K_FILES_]
        double * vector_perturbations_data[_MAX_NUMBER_OF_K_FILES_]
        double * tensor_perturbations_data[_MAX_NUMBER_OF_K_FILES_]
        int size_scalar_perturbation_data[_MAX_NUMBER_OF_K_FILES_]
        int size_vector_perturbation_data[_MAX_NUMBER_OF_K_FILES_]
        int size_tensor_perturbation_data[_MAX_NUMBER_OF_K_FILES_]

        #double * alpha_idm_dr
        #double * beta_idr


    cdef struct transfer:
        ErrorMsg error_message


    cdef struct primordial:
        ErrorMsg error_message
        double k_pivot
        double A_s
        double n_s
        double alpha_s
        double beta_s
        double r
        double n_t
        double alpha_t
        double V0
        double V1
        double V2
        double V3
        double V4
        double f_cdi
        double n_cdi
        double c_ad_cdi
        double n_ad_cdi
        double f_nid
        double n_nid
        double c_ad_nid
        double n_ad_nid
        double f_niv
        double n_niv
        double c_ad_niv
        double n_ad_niv
        double phi_min
        double phi_max
        int lnk_size
        int * ic_ic_size


    cdef struct harmonic:
        ErrorMsg error_message
        int has_tt
        int has_te
        int has_ee
        int has_bb
        int has_pp
        int has_tp
        int has_ep
        int has_dd
        int has_td
        int has_ll
        int has_dl
        int has_tl
        int l_max_tot
        int ** l_max_ct
        int ct_size
        int * ic_size
        int * ic_ic_size
        int md_size
        int d_size
        int non_diag
        int index_ct_tt
        int index_ct_te
        int index_ct_ee
        int index_ct_bb
        int index_ct_pp
        int index_ct_tp
        int index_ct_ep
        int index_ct_dd
        int index_ct_td
        int index_ct_pd
        int index_ct_ll
        int index_ct_dl
        int index_ct_tl
        int * l_size
        int index_md_scalars


    cdef struct output:
        ErrorMsg error_message


    cdef struct distortions:
        double * sd_parameter_table
        int index_type_g
        int index_type_mu
        int index_type_y
        int index_type_PCA
        int type_size
        double * DI
        double * x
        double DI_units
        double x_to_nu
        int x_size
        ErrorMsg error_message


    cdef struct lensing:
        int has_tt
        int has_ee
        int has_te
        int has_bb
        int has_pp
        int has_tp
        int has_dd
        int has_td
        int has_ll
        int has_dl
        int has_tl
        int index_lt_tt
        int index_lt_te
        int index_lt_ee
        int index_lt_bb
        int index_lt_pp
        int index_lt_tp
        int index_lt_dd
        int index_lt_td
        int index_lt_ll
        int index_lt_dl
        int index_lt_tl
        int * l_max_lt
        int lt_size
        int has_lensed_cls
        int l_lensed_max
        int l_unlensed_max
        ErrorMsg error_message


    cdef struct fourier:
        short has_pk_matter
        short has_pk_m
        short has_pk_cb
        int method
        int ic_size
        int ic_ic_size
        int k_size
        int ln_tau_size
        int tau_size
        int index_tau_min_nl
        double * k
        double * ln_k
        double * ln_tau
        double * tau
        double ** ln_pk_l
        double ** ln_pk_nl
        double * sigma8
        int index_pk_m
        int index_pk_cb
        int index_pk_total
        int index_pk_cluster
        int index_md_scalars
        short * is_non_zero
        ErrorMsg error_message


    cdef struct file_content:
        char * filename
        int size
        FileArg * name
        FileArg * value
        short * read

    void parser_free(void*)
    void lensing_free(void*)
    void harmonic_free(void*)
    void transfer_free(void*)
    void primordial_free(void*)
    void perturbations_free(void*)
    void thermodynamics_free(void*)
    void background_free(void*)
    void fourier_free(void*)
    void distortions_free(void*)

    cdef int _FAILURE_
    cdef int _FALSE_
    cdef int _TRUE_

    int input_read_from_file(void*, void*, void*, void*, void*, void*, void*, void*, void*,
        void*, void*, void*, ErrorMsg errmsg) nogil
    int background_init(void*,void*) nogil
    int thermodynamics_init(void*,void*,void*) nogil
    int perturbations_init(void*,void*,void*,void*) nogil
    int primordial_init(void*,void*,void*) nogil
    int fourier_init(void*,void*,void*,void*,void*,void*) nogil
    int transfer_init(void*,void*,void*,void*,void*,void*) nogil
    int harmonic_init(void*,void*,void*,void*,void*,void*,void*) nogil
    int lensing_init(void*,void*,void*,void*,void*) nogil
    int distortions_init(void*,void*,void*,void*,void*,void*) nogil

    int background_tau_of_z(void* pba, double z, double* tau) nogil
    int background_z_of_tau(void* pba, double tau, double* z) nogil
    int background_at_z(void* pba, double z, int return_format, int inter_mode, int * last_index, double *pvecback) nogil
    int background_at_tau(void* pba, double tau, int return_format, int inter_mode, int * last_index, double *pvecback) nogil
    int background_output_titles(void * pba, char titles[_MAXTITLESTRINGLENGTH_]) nogil
    int background_output_data(void *pba, int number_of_titles, double *data) nogil

    int thermodynamics_at_z(void * pba, void * pth, double z, int inter_mode, int * last_index, double *pvecback, double *pvecthermo) nogil
    int thermodynamics_output_titles(void * pba, void *pth, char titles[_MAXTITLESTRINGLENGTH_]) nogil
    int thermodynamics_output_data(void *pba, void *pth, int number_of_titles, double *data) nogil

    int perturbations_output_data_at_z(void *pba,void *ppt, file_format output_format, double z, int number_of_titles, double *data) nogil
    int perturbations_output_firstline_and_ic_suffix(void *ppt, int index_ic, char first_line[_LINE_LENGTH_MAX_], FileName ic_suffix) nogil
    int perturbations_output_titles(void *pba, void *ppt,  file_format output_format, char titles[_MAXTITLESTRINGLENGTH_]) nogil

    int primordial_output_titles(void * ppt, void *ppm, char titles[_MAXTITLESTRINGLENGTH_]) nogil
    int primordial_output_data(void *ppt, void *ppm, int number_of_titles, double *data) nogil
    int primordial_spectrum_at_k(void * ppm, int index_md, linear_or_logarithmic mode, double k, double * pk) nogil

    int harmonic_cl_at_l(void* phr,double l,double * cl,double * * cl_md,double * * cl_md_ic) nogil
    int lensing_cl_at_l(void * ple,int l,double * cl_lensed) nogil

    int harmonic_pk_at_z(
        void * pba,
        void * phr,
        int mode,
        double z,
        double * output_tot,
        double * output_ic,
        double * output_cb_tot,
        double * output_cb_ic
        ) nogil

    int harmonic_pk_at_k_and_z(
        void* pba,
        void * ppm,
        void * phr,
        double k,
        double z,
        double * pk,
        double * pk_ic,
        double * pk_cb,
        double * pk_cb_ic) nogil

    int harmonic_pk_nl_at_k_and_z(
        void* pba,
        void * ppm,
        void * phr,
        double k,
        double z,
        double * pk,
        double * pk_cb) nogil

    int harmonic_pk_nl_at_z(
        void * pba,
        void * phr,
        int mode,
        double z,
        double * output_tot,
        double * output_cb_tot) nogil

    int fourier_pk_at_k_and_z(
        void * pba,
        void * ppm,
        void * pfo,
        int pk_output,
        double k,
        double z,
        int index_pk,
        double * out_pk,
        double * out_pk_ic) nogil

    int fourier_pk_tilt_at_k_and_z(
        void * pba,
        void * ppm,
        void * pfo,
        int pk_output,
        double k,
        double z,
        int index_pk,
        double * pk_tilt) nogil

    int fourier_sigmas_at_z(
        void * ppr,
        void * pba,
        void * pfo,
        double R,
        double z,
        int index_pk,
        int sigma_output,
        double * result) nogil

    int fourier_pks_at_kvec_and_zvec(
        void * pba,
        void * pfo,
        int pk_output,
        double * kvec,
        int kvec_size,
        double * zvec,
        int zvec_size,
        double * out_pk,
        double * out_pk_cb) nogil

    int fourier_get_source(
        void * pba,
        void * ppt,
        void * pfo,
        int index_k,
        int index_ic,
        int index_tp,
        int index_tau,
        double ** sources,
        double * source) nogil

    int fourier_hmcode_sigma8_at_z(void* pba, void* pfo, double z, double* sigma_8, double* sigma_8_cb) nogil
    int fourier_hmcode_sigmadisp_at_z(void* pba, void* pfo, double z, double* sigma_disp, double* sigma_disp_cb) nogil
    int fourier_hmcode_sigmadisp100_at_z(void* pba, void* pfo, double z, double* sigma_disp_100, double* sigma_disp_100_cb) nogil
    int fourier_hmcode_sigmaprime_at_z(void* pba, void* pfo, double z, double* sigma_prime, double* sigma_prime_cb) nogil
    int fourier_hmcode_window_nfw(void* pfo, double k, double rv, double c, double* window_nfw) nogil

    int fourier_k_nl_at_z(void* pba, void* pfo, double z, double* k_nl, double* k_nl_cb) nogil

    int harmonic_firstline_and_ic_suffix(void *ppt, int index_ic, char first_line[_LINE_LENGTH_MAX_], FileName ic_suffix) nogil

    int harmonic_fast_pk_at_kvec_and_zvec(
                  void * pba,
                  void * phr,
                  double * kvec,
                  int kvec_size,
                  double * zvec,
                  int zvec_size,
                  double * pk_tot_out,
                  double * pk_cb_tot_out,
                  int nonlinear) nogil
