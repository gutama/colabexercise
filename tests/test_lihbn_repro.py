from pathlib import Path
import numpy as np
import pytest
from lihbn_repro.wannier import (make_demo_hr,parse_hr,one_particle_eigenpairs,
                                 sector_crosscheck,spin_blocks,spin_blocked_to_interleaved)
from lihbn_repro.qe_audit import audit


def test_valid_hr_and_gamma_hermiticity(tmp_path):
    path=make_demo_hr(tmp_path/'demo_hr.dat',nw=11)
    data=parse_hr(path,expected_wann=11,expected_nrpts=3)
    assert data.num_wann==11
    assert data.max_hr_hermiticity<1e-14
    for k in [(0.,0.,0.),(0.25,0.1,0.)]:
        mat=data.h_at_k(k)
        np.testing.assert_allclose(mat,mat.conj().T,atol=1e-12)
    assert np.max(np.abs(data.h_at_k()-data.onsite()))>0.001
    u,v=spin_blocks(data,data,up_labels=range(1,12),down_labels=range(1,12))
    assert u.shape==v.shape==(11,11)


def test_duplicate_hr_fails_closed(tmp_path):
    path=make_demo_hr(tmp_path/'broken.dat',nw=2)
    lines=path.read_text().splitlines()
    lines[-1]=lines[-2]
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='Duplicate'):
        parse_hr(path)


def test_nonhermitian_hr_rejected(tmp_path):
    path=make_demo_hr(tmp_path/'broken.dat',nw=2)
    rows=path.read_text().splitlines()
    rows[4]=rows[4].rsplit(' ',2)[0]+' 1.0 2.0'  # mutate H(R=0)[0,0]
    path.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='Hermiticity'):
        parse_hr(path)


def test_quadratic_sector_state_fidelity():
    a=np.array([[0.2,-1j],[1j,-0.5]],complex)
    b=np.array([[0.8,0.3],[0.3,-0.4]],complex)
    r=sector_crosscheck(a,b,n_up=1,n_down=1)
    assert r['dimension']==4 and not r['skipped']
    assert r['energy_error_ev']<1e-12
    assert r['slater_residual_ev']<1e-12
    assert r['eigenstate_fidelity']>1-1e-12
    assert abs(r['slater_norm']-1)<1e-12
    w=one_particle_eigenpairs(a,1)
    assert w['max_residual_ev']<1e-12
    assert w['idempotency_defect']<1e-12


def test_larger_sector_checks_nontrivial_slater():
    rng=np.random.default_rng(4)
    a=rng.normal(size=(3,3)); a=(a+a.T)/2
    b=rng.normal(size=(3,3)); b=(b+b.T)/2
    r=sector_crosscheck(a,b,n_up=2,n_down=1)
    assert r['dimension']==9
    assert r['energy_error_ev']<1e-11
    assert r['eigenstate_fidelity']>1-1e-10
    assert sector_crosscheck(a,b,n_up=2,n_down=1,max_dimension=2)['skipped']


def test_qe_log_inner_not_completion(tmp_path):
    log=tmp_path/'pw.x.out'
    log.write_text('iteration # 2\n! total energy = -411.98010436 Ry\n'
                   'convergence has been achieved in 20 iterations\n'
                   'Using ACE for calculation of exact exchange\n'
                   'Fock energy (ACE) = -109.59347970 Ry\n')
    d=audit(log)
    assert not d['job_done']
    assert d['counts']['inner_scf_convergences']==1
    assert d['last_ace_fock_energy_ry']==pytest.approx(-109.59347970)


def test_eig_csv_index_and_energy_validation(tmp_path):
    pandas=pytest.importorskip("pandas")
    from lihbn_repro.validation import compare_eig_csv
    import pandas as pd
    rows=[]
    for sp in (1,2):
        for k in (1,2):
            for b in (1,2):
                rows.append((k,sp,b,float(-sp*k+b)))
    pd.DataFrame(rows,columns=['k_index','spin','band','energy_eV']).to_csv(tmp_path/'spectrum.csv',index=False)
    for sp,name in [(1,'up'),(2,'down')]:
        array=np.array([(b,k,-sp*k+b) for k in (1,2) for b in (1,2)],float)
        np.savetxt(tmp_path/f'{name}.eig',array)
    report=compare_eig_csv(tmp_path/'spectrum.csv',tmp_path/'up.eig',tmp_path/'down.eig',expected_kpoints=2,expected_bands=2)
    assert report['up']['max_abs_ev']==0 and report['down']['max_abs_ev']==0


def test_asymmetric_spin_order_permutation():
    """Spin-resolved blocks cannot silently collapse to a common spin Hamiltonian."""
    up = np.array([[0.3, 0.4j], [-0.4j, 1.7]], complex)
    down = np.array([[-2., 0.6], [0.6, 0.2]], complex)
    got = spin_blocked_to_interleaved(up, down)
    blocked = np.block([[up, np.zeros_like(up)], [np.zeros_like(down), down]])
    order = [0, 2, 1, 3]
    np.testing.assert_allclose(got, blocked[np.ix_(order, order)], atol=0, rtol=0)
    np.testing.assert_allclose(got[0::2, 0::2], up)
    np.testing.assert_allclose(got[1::2, 1::2], down)
    assert not np.allclose(got[0::2, 0::2], got[1::2, 1::2])
    with pytest.raises(ValueError):
        spin_blocked_to_interleaved(up, np.eye(3))


def test_eig_csv_rejects_matching_but_incomplete_grid(tmp_path):
    """Two mutually consistent corrupted inputs must fail grid coverage validation."""
    import pandas as pd
    from lihbn_repro.validation import compare_eig_csv
    rows = [(k, spin, b, -spin*k + b)
            for spin in (1, 2) for k in (2, 3) for b in (1, 2)]
    pd.DataFrame(rows, columns=['k_index','spin','band','energy_eV']).to_csv(
        tmp_path/'wrong_grid.csv', index=False)
    for spin, name in ((1,'up'),(2,'down')):
        lines = [(b, k, -spin*k + b) for k in (2,3) for b in (1,2)]
        np.savetxt(tmp_path/f'{name}.eig', np.array(lines))
    with pytest.raises(ValueError, match='missing or unexpected'):
        compare_eig_csv(tmp_path/'wrong_grid.csv', tmp_path/'up.eig',
                        tmp_path/'down.eig', expected_kpoints=2, expected_bands=2)


def test_colab_notebook_uses_published_github_by_default():
    import json
    path = Path(__file__).resolve().parents[1]/'notebooks/lihbn_reproduction.ipynb'
    cells = json.loads(path.read_text())['cells']
    assert 'SOURCE_MODE = "GITHUB"' in ''.join(cells[1]['source'])
