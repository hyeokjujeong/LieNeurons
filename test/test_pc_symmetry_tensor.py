import torch

from experiment.pc_se3_congruence.data_synth import (c2_clouds,
                                                     contact_spring_all_pairs_K,
                                                     contact_spring_kernel_K)
from experiment.pc_se3_congruence.encoders import WrenchEdgeEncoder
from experiment.pc_se3_congruence.models import WrenchSecondMomentModel
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)


def test_analytic_second_moment_matches_all_pair_target():
    gen = torch.Generator().manual_seed(11)
    P = c2_clouds(3, 24, gen, dtype=torch.float64)
    sigma = 0.7
    model = WrenchSecondMomentModel(
        WrenchEdgeEncoder(graph='all'),
        weight_mode='analytic', sigma=sigma)
    actual = model(P)
    expected = contact_spring_all_pairs_K(P, sigma_k=sigma)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_all_pair_second_moment_is_full_rank_and_equivariant_on_c2():
    gen = torch.Generator().manual_seed(23)
    P = c2_clouds(4, 24, gen, dtype=torch.float64)
    model = WrenchSecondMomentModel(
        WrenchEdgeEncoder(graph='all'), weight_mode='uniform')

    K = model(P)
    assert (torch.linalg.eigvalsh(K)[:, 0] > 1e-10).all()

    perm = torch.randperm(P.shape[1], generator=gen)
    torch.testing.assert_close(model(P[:, perm]), K, rtol=1e-13, atol=1e-13)

    R, p = random_SE3(1.0, gen, dtype=torch.float64)
    rho = coadjoint(R, p)
    transformed = model(transform_cloud(P, R, p))
    expected = rho @ K @ rho.transpose(-1, -2)
    assert scaled_err(transformed, expected) < 1e-12


def test_analytic_second_moment_matches_local_kernel_target():
    gen = torch.Generator().manual_seed(31)
    P = c2_clouds(3, 24, gen, dtype=torch.float64)
    sigma = 0.7
    candidates = 12
    model = WrenchSecondMomentModel(
        WrenchEdgeEncoder(graph='kernel', candidate_k=candidates),
        weight_mode='analytic', sigma=sigma)
    actual = model(P)
    expected = contact_spring_kernel_K(
        P, candidate_k=candidates, sigma_k=sigma)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_local_kernel_tensor_is_permutation_invariant_and_equivariant_on_c2():
    gen = torch.Generator().manual_seed(37)
    P = c2_clouds(4, 24, gen, dtype=torch.float64)
    model = WrenchSecondMomentModel(
        WrenchEdgeEncoder(graph='kernel', candidate_k=12),
        weight_mode='analytic', sigma=0.7)

    K = model(P)
    assert (torch.linalg.eigvalsh(K)[:, 0] > 1e-10).all()

    perm = torch.randperm(P.shape[1], generator=gen)
    torch.testing.assert_close(model(P[:, perm]), K, rtol=1e-11, atol=1e-12)

    R, p = random_SE3(1.0, gen, dtype=torch.float64)
    rho = coadjoint(R, p)
    transformed = model(transform_cloud(P, R, p))
    expected = rho @ K @ rho.transpose(-1, -2)
    assert scaled_err(transformed, expected) < 1e-11
