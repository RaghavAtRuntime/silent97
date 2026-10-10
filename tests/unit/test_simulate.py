"""Tests for the synthetic customer simulator, including Gate G1 checks."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from silent97.data import schema, simulate
from silent97.data.simulate import SimConfig

SMALL_N = 20_000


def _pop(seed: int = 0, **overrides: object) -> simulate.Population:
    cfg = SimConfig(**{"n_customers": SMALL_N, **overrides})  # type: ignore[arg-type]
    return simulate.simulate_population(cfg, np.random.default_rng(seed))


def _col(pop: simulate.Population, name: str) -> np.ndarray:
    return pop.table.column(name).to_numpy(zero_copy_only=False)


@pytest.fixture(scope="module")
def small() -> simulate.Population:
    return _pop(0)


@pytest.fixture(scope="module")
def full() -> simulate.Population:
    return simulate.simulate_population(SimConfig(), np.random.default_rng(12345))


# --------------------------------------------------------------------------- config


class TestConfig:
    def test_defaults(self) -> None:
        cfg = SimConfig()
        assert cfg.n_customers == 200_000
        assert cfg.target_nps == 20.0
        assert len(cfg.base_score_probs) == 11

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"signal_strength": -0.1}, "signal_strength"),
            ({"signal_strength": 1.1}, "signal_strength"),
            ({"response_rate": 0.0}, "response_rate"),
            ({"response_rate": 1.0}, "response_rate"),
            ({"n_customers": 10}, "n_customers"),
            ({"target_nps": 100.0}, "target_nps"),
            ({"sigma_idiosyncratic": -1.0}, "sigma_idiosyncratic"),
            ({"sd_segment": -0.1}, "sd_segment"),
            ({"mnar_strength": -0.5}, "mnar_strength"),
            ({"base_score_probs": (0.5, 0.5)}, "base_score_probs"),
            ({"base_score_probs": (0.0,) + (0.1,) * 10}, "base_score_probs"),
            ({"beta": {"wealth": 1.0}}, "beta"),
            ({"mar_coefs": {"age": 1.0}}, "mar_coefs"),
        ],
    )
    def test_rejects_bad_values(self, overrides: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SimConfig(**overrides)  # type: ignore[arg-type]

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            SimConfig.from_dict({"n_customer": 5})

    def test_load_config_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "sim.yaml"
        path.write_text(
            "seed: 7\nout_dir: out/sim\nsimulation:\n  n_customers: 5000\n  signal_strength: 0.3\n"
        )
        cfg, seed, out_dir = simulate.load_config(path)
        assert seed == 7
        assert out_dir == Path("out/sim")
        assert cfg.n_customers == 5000
        assert cfg.signal_strength == 0.3
        assert cfg.target_nps == SimConfig().target_nps

    def test_from_dict_accepts_list_probs(self) -> None:
        probs = list(SimConfig().base_score_probs)
        assert SimConfig.from_dict({"base_score_probs": probs}).base_score_probs == tuple(probs)

    @pytest.mark.parametrize(
        "text, match",
        [
            ("- 1\n- 2\n", "mapping"),
            ("seed: 1\nsimulations: {}\n", "unknown"),
            ("seed: -1\n", "seed"),
        ],
    )
    def test_load_config_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        path = tmp_path / "sim.yaml"
        path.write_text(text)
        with pytest.raises(ValueError, match=match):
            simulate.load_config(path)

    def test_load_config_requires_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "sim.yaml"
        path.write_text("simulation:\n  n_customers: 5000\n")
        with pytest.raises(ValueError, match="seed"):
            simulate.load_config(path)

    def test_repo_config_loads(self) -> None:
        cfg, seed, _ = simulate.load_config(Path("configs/simulate.yaml"))
        assert cfg.n_customers == 200_000
        assert isinstance(seed, int)


# --------------------------------------------------------------------------- structure


class TestStructure:
    def test_schema(self, small: simulate.Population) -> None:
        assert small.table.schema.equals(schema.SCHEMA)
        assert small.table.num_rows == SMALL_N

    def test_ids(self, small: simulate.Population) -> None:
        np.testing.assert_array_equal(_col(small, "customer_id"), np.arange(SMALL_N))

    def test_covariate_ranges(self, small: simulate.Population) -> None:
        assert set(_col(small, "age_band")) == set(simulate.AGE_BANDS)
        assert set(_col(small, "province")) == set(simulate.PROVINCES)
        assert set(_col(small, "tenure_band")) <= set(simulate.TENURE_BANDS)
        tenure = _col(small, "tenure_years")
        assert tenure.min() >= 0
        digital = _col(small, "digital_engagement")
        assert digital.min() >= 0 and digital.max() <= 1
        assert _col(small, "complaints_past_year").min() >= 0
        assert _col(small, "monthly_txn_count").min() >= 0
        for col in ("has_credit_card", "has_chequing", "has_wealth", "has_insurance"):
            share = _col(small, col).mean()
            assert 0.02 < share < 0.98, col

    def test_tenure_bounded_by_adult_years(self, small: simulate.Population) -> None:
        # Customers 18-24 cannot have more than 7 years of tenure.
        young = _col(small, "age_band") == "18-24"
        assert _col(small, "tenure_years")[young].max() <= 7.0

    def test_tenure_band_consistent(self, small: simulate.Population) -> None:
        tenure = _col(small, "tenure_years")
        band = _col(small, "tenure_band")
        expected = np.array(simulate.TENURE_BANDS)[np.digitize(tenure, simulate.TENURE_EDGES)]
        np.testing.assert_array_equal(band, expected)

    def test_segments(self, small: simulate.Population) -> None:
        seg = _col(small, "segment")
        expected = np.char.add(
            np.char.add(_col(small, "province").astype(str), "|"),
            _col(small, "age_band").astype(str),
        )
        np.testing.assert_array_equal(seg.astype(str), expected)
        seg_id = _col(small, "segment_id")
        assert seg_id.min() >= 0 and seg_id.max() < 60
        # segment_id and segment label are a bijection
        pairs = set(zip(seg_id.tolist(), seg.tolist(), strict=True))
        assert len(pairs) == len(set(seg_id.tolist()))

    def test_scores_and_indicators(self, small: simulate.Population) -> None:
        score = _col(small, "score")
        assert score.min() >= 0 and score.max() <= 10
        np.testing.assert_array_equal(_col(small, "promoter"), score >= 9)
        np.testing.assert_array_equal(_col(small, "detractor"), score <= 6)

    def test_scores_follow_thresholds(self, small: simulate.Population) -> None:
        thresholds = np.array(small.manifest["derived"]["thresholds"])
        ystar = _col(small, "latent_response")
        np.testing.assert_array_equal(
            _col(small, "score"), np.searchsorted(thresholds, ystar, side="left")
        )

    def test_every_score_occurs(self, full: simulate.Population) -> None:
        assert set(np.unique(_col(full, "score"))) == set(range(11))

    def test_no_missing_values(self, small: simulate.Population) -> None:
        for name in small.table.column_names:
            assert small.table.column(name).null_count == 0, name


# --------------------------------------------------------------------------- ground truth


class TestTruth:
    def test_expected_nps_hits_target(self, full: simulate.Population) -> None:
        assert full.manifest["truth"]["expected_nps"] == pytest.approx(20.0, abs=1e-6)

    def test_manifest_truth_matches_table(self, full: simulate.Population) -> None:
        prom = _col(full, "promoter").mean()
        det = _col(full, "detractor").mean()
        truth = full.manifest["truth"]
        assert truth["nps"] == pytest.approx(100 * (prom - det), abs=1e-12)
        assert truth["promoter_share"] == pytest.approx(prom, abs=1e-12)
        assert truth["detractor_share"] == pytest.approx(det, abs=1e-12)

    def test_segment_truth(self, full: simulate.Population) -> None:
        seg_truth = full.manifest["truth"]["segment_nps"]
        assert len(seg_truth) == 60
        seg = _col(full, "segment").astype(str)
        value = 100.0 * (
            _col(full, "promoter").astype(float) - _col(full, "detractor").astype(float)
        )
        for label in ("ON|35-44", "PE|18-24"):
            assert seg_truth[label]["nps"] == pytest.approx(value[seg == label].mean())
            assert seg_truth[label]["n"] == int((seg == label).sum())

    def test_expected_nps_function(self) -> None:
        eta = np.array([0.0, 0.0])
        thresholds = simulate.base_cutpoints(SimConfig().base_score_probs)
        probs = np.array(SimConfig().base_score_probs)
        expected = 100 * (probs[9:].sum() - probs[:7].sum())
        assert simulate.expected_nps(eta, thresholds) == pytest.approx(expected)

    def test_solve_shift(self) -> None:
        rng = np.random.default_rng(3)
        eta = rng.normal(0.4, 1.2, size=5000)
        base = simulate.base_cutpoints(SimConfig().base_score_probs)
        for target in (-30.0, 0.0, 20.0, 55.0):
            shift = simulate.solve_cutpoint_shift(eta, base, target)
            assert simulate.expected_nps(eta, base + shift) == pytest.approx(target, abs=1e-8)

    def test_complaints_lower_satisfaction(self, full: simulate.Population) -> None:
        comp = _col(full, "complaints_past_year")
        latent = _col(full, "latent")
        assert latent[comp > 0].mean() < latent[comp == 0].mean() - 0.3


# --------------------------------------------------------------------------- response


class TestResponse:
    def test_propensities_valid(self, small: simulate.Population) -> None:
        for m in schema.MECHANISMS:
            p = _col(small, f"propensity_{m}")
            assert p.min() > 0 and p.max() < 1, m
            assert p.mean() == pytest.approx(0.03, abs=1e-9), m

    def test_mcar_constant(self, small: simulate.Population) -> None:
        p = _col(small, "propensity_mcar")
        assert np.all(p == p[0])

    def test_realized_rate(self, full: simulate.Population) -> None:
        for m in schema.MECHANISMS:
            rate = _col(full, f"responded_{m}").mean()
            # 200k Bernoulli(~0.03): sd of rate ~ 0.0004
            assert rate == pytest.approx(0.03, abs=0.002), m

    def test_mar_ignores_latent(self) -> None:
        rng = np.random.default_rng(0)
        index = rng.normal(size=1000)
        z1, z2 = rng.normal(size=1000), rng.normal(size=1000)
        for mech in ("mcar", "mar"):
            a = simulate.response_propensity(
                index, z1, mechanism=mech, response_rate=0.05, mnar_strength=0.7
            )
            b = simulate.response_propensity(
                index, z2, mechanism=mech, response_rate=0.05, mnar_strength=0.7
            )
            np.testing.assert_array_equal(a, b)

    def test_mnar_is_u_shaped(self) -> None:
        index = np.zeros(3)
        z = np.array([-2.0, 0.0, 2.0])
        p = simulate.response_propensity(
            index, z, mechanism="mnar", response_rate=0.05, mnar_strength=0.5
        )
        assert p[0] == pytest.approx(p[2])
        assert p[0] > p[1]

    def test_mnar_in_population(self, full: simulate.Population) -> None:
        latent = _col(full, "latent")
        p = _col(full, "propensity_mnar")
        lo, mid_lo, mid_hi, hi = np.quantile(latent, [0.1, 0.45, 0.55, 0.9])
        middle = p[(latent > mid_lo) & (latent < mid_hi)].mean()
        assert p[latent < lo].mean() > 1.5 * middle
        assert p[latent > hi].mean() > 1.5 * middle

    def test_unknown_mechanism(self) -> None:
        with pytest.raises(ValueError, match="mechanism"):
            simulate.response_propensity(
                np.zeros(2), np.zeros(2), mechanism="nmar", response_rate=0.1, mnar_strength=0
            )

    def test_mar_and_mnar_shift_respondent_nps(self, full: simulate.Population) -> None:
        resp = full.manifest["response"]
        assert resp["mcar"]["expected_respondent_nps"] == pytest.approx(
            full.manifest["truth"]["nps"], abs=1e-9
        )
        truth = full.manifest["truth"]["nps"]
        # MAR: customers who look satisfied on observed traits respond more.
        assert resp["mar"]["expected_respondent_nps"] > truth + 1.0
        # MNAR adds a U-shaped dependence on unexplained satisfaction on top of MAR.
        assert (
            abs(resp["mnar"]["expected_respondent_nps"] - resp["mar"]["expected_respondent_nps"])
            > 1.0
        )


# --------------------------------------------------------------------------- signal knob


class TestSignal:
    def test_zero_strength_is_noise(self) -> None:
        pop = _pop(1, signal_strength=0.0)
        r = np.corrcoef(_col(pop, "signal"), _col(pop, "score"))[0, 1]
        assert abs(r) < 0.03

    def test_full_strength_orders_scores(self) -> None:
        pop = _pop(1, signal_strength=1.0)
        order = np.argsort(_col(pop, "signal"), kind="stable")
        assert np.all(np.diff(_col(pop, "score")[order]) >= 0)

    def test_monotone_in_strength(self) -> None:
        corrs = [
            _pop(2, signal_strength=a).manifest["signal"]["corr_score"]
            for a in (0.0, 0.3, 0.6, 0.9)
        ]
        assert all(b > a for a, b in itertools.pairwise(corrs))
        assert corrs[-1] > 0.8

    def test_signal_standardized(self, small: simulate.Population) -> None:
        s = _col(small, "signal")
        assert s.mean() == pytest.approx(0.0, abs=0.05)
        assert s.std() == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- reproducibility


class TestReproducibility:
    def test_same_seed_same_table(self) -> None:
        assert _pop(5).table.equals(_pop(5).table)

    def test_different_seed_differs(self) -> None:
        a, b = _pop(5), _pop(6)
        assert not np.array_equal(_col(a, "score"), _col(b, "score"))

    def test_signal_knob_leaves_population_unchanged(self) -> None:
        a, b = _pop(9, signal_strength=0.2), _pop(9, signal_strength=0.8)
        for name in schema.ALL_COLUMNS:
            if name == "signal":
                assert not np.array_equal(_col(a, name), _col(b, name))
            else:
                np.testing.assert_array_equal(_col(a, name), _col(b, name), err_msg=name)

    def test_response_knobs_leave_population_unchanged(self) -> None:
        a = _pop(9)
        b = _pop(9, mnar_strength=1.0, response_rate=0.05)
        response = set(schema.RESPONSE_COLUMNS)
        for name in schema.ALL_COLUMNS:
            if name not in response:
                np.testing.assert_array_equal(_col(a, name), _col(b, name), err_msg=name)
        assert not np.array_equal(_col(a, "propensity_mnar"), _col(b, "propensity_mnar"))

    def test_byte_identical_parquet(self, tmp_path: Path) -> None:
        """Gate G1: identical seeds produce byte-identical Parquet (and manifest)."""
        cfg = SimConfig()
        paths = [simulate.run(cfg, seed=2026, out_dir=tmp_path / f"run{i}") for i in (0, 1)]
        (pq_a, man_a), (pq_b, man_b) = paths
        assert pq_a.read_bytes() == pq_b.read_bytes()
        assert man_a.read_bytes() == man_b.read_bytes()

    def test_written_files(self, tmp_path: Path) -> None:
        cfg = SimConfig(n_customers=SMALL_N)
        parquet_path, manifest_path = simulate.run(cfg, seed=3, out_dir=tmp_path)
        table = pq.read_table(parquet_path)
        assert table.schema.equals(schema.SCHEMA)
        assert table.equals(simulate.simulate_population(cfg, np.random.default_rng(3)).table)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == 3
        assert manifest["config"]["n_customers"] == SMALL_N
        digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        assert manifest["files"]["parquet"]["sha256"] == digest
        assert manifest["files"]["parquet"]["n_rows"] == SMALL_N

    def test_cli(self, tmp_path: Path) -> None:
        code = simulate.main(
            [
                "--config",
                "configs/simulate.yaml",
                "--seed",
                "4",
                "--n-customers",
                "5000",
                "--out",
                str(tmp_path),
            ]
        )
        assert code == 0
        assert pq.read_table(tmp_path / "population.parquet").num_rows == 5000

    def test_cli_uses_config_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "sim.yaml"
        cfg.write_text(f"seed: 11\nout_dir: {tmp_path / 'out'}\nsimulation:\n  n_customers: 2000\n")
        assert simulate.main(["--config", str(cfg)]) == 0
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["seed"] == 11
        assert manifest["config"]["n_customers"] == 2000


def test_gate_g1_realized_nps_within_one_point_across_20_seeds() -> None:
    """Gate G1: realized NPS of a 200,000-customer population within +/-1 of +20."""
    cfg = SimConfig()
    realized = []
    for seed in range(20):
        pop = simulate.simulate_population(cfg, np.random.default_rng(seed))
        realized.append(pop.manifest["truth"]["nps"])
    realized_arr = np.array(realized)
    assert np.all(np.abs(realized_arr - 20.0) <= 1.0), realized_arr
