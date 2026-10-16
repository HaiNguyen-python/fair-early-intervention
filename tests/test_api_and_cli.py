"""API contract with a real trained bundle, and the CLI end to end on a small dataset."""

import pytest
from fastapi.testclient import TestClient

from fair_intervention import serve
from fair_intervention.agents import QLearningAgent, ThresholdPolicy
from fair_intervention.cli import main
from fair_intervention.env import EnvConfig
from fair_intervention.risk import fit_bundle

STUDENT = {
    "clicks_total": 40,
    "clicks_last_week": 0,
    "clicks_trend": -2.0,
    "active_days": 4,
    "distinct_sites": 3,
    "days_since_last_click": 12,
    "clicks_before_start": 0,
    "n_submitted": 0,
    "n_missed": 2,
    "n_late": 0,
    "mean_score": None,
    "num_of_prev_attempts": 1,
    "studied_credits": 120,
}
ENGAGED = {
    **STUDENT,
    "clicks_total": 900,
    "clicks_last_week": 150,
    "clicks_trend": 0.1,
    "active_days": 30,
    "distinct_sites": 25,
    "days_since_last_click": 0,
    "n_submitted": 2,
    "n_missed": 0,
    "mean_score": 85,
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
}


@pytest.fixture(scope="module")
def bundle(splits):
    return fit_bundle(splits, 0.2, "test")


@pytest.fixture()
def client(bundle):
    serve.app.dependency_overrides[serve.get_bundle] = lambda: bundle
    serve.app.dependency_overrides[serve.get_policy] = ThresholdPolicy
    yield TestClient(serve.app)
    serve.app.dependency_overrides.clear()


def test_score_contract_and_ordering(client):
    risky = client.post("/score", json={"week": 6, "features": STUDENT}).json()
    safe = client.post("/score", json={"week": 6, "features": ENGAGED}).json()
    assert set(risky) == {
        "week",
        "risk",
        "on_contact_list",
        "risk_bucket",
        "top_reasons",
        "recommended_action",
        "policy",
    }
    assert 0 <= safe["risk"] < risky["risk"] <= 1
    assert risky["on_contact_list"] and not safe["on_contact_list"]
    assert risky["recommended_action"] in {"email", "call"}
    assert safe["recommended_action"] == "none"
    assert risky["top_reasons"] and all(r["contribution"] > 0 for r in risky["top_reasons"])
    assert risky["policy"] == "Threshold rule"


def test_score_rejects_bad_input(client):
    assert client.post("/score", json={"week": 5, "features": STUDENT}).status_code == 422
    bad = {**STUDENT, "clicks_total": -1}
    assert client.post("/score", json={"week": 6, "features": bad}).status_code == 422
    assert client.get("/health").json()["status"] == "ok"


def test_policy_observation_maps_request_to_simulator_state():
    obs = serve.policy_observation(0.85, 8, 3.0, "0-10%", 5)
    assert obs.tolist() == [4, 4, 0, 1, 2]
    assert serve.policy_observation(0.1, 4, None, None, 0).tolist()[:4] == [0, 0, 3, 0]


def test_cli_generate_train_then_serve_from_artifacts(tmp_path, monkeypatch, capsys):
    data = tmp_path / "raw"
    main(["generate", "--out", str(data), "--students", "150", "--seed", "4"])
    art = tmp_path / "artifacts"
    main(
        [
            "train",
            "--source",
            "oulad",
            "--data-dir",
            str(data),
            "--artifacts",
            str(art),
            "--episodes",
            "50",
        ]
    )
    assert "thresholds" in capsys.readouterr().out
    monkeypatch.setenv("FAIR_INTERVENTION_ARTIFACTS", str(art))
    serve.get_bundle.cache_clear()
    serve.get_policy.cache_clear()
    assert isinstance(serve.get_policy(), QLearningAgent)
    response = TestClient(serve.app).post(
        "/score", json={"week": 4, "features": STUDENT, "imd_band": "0-10%"}
    )
    assert response.status_code == 200 and response.json()["policy"] == "Q-learning"
    serve.get_bundle.cache_clear()
    serve.get_policy.cache_clear()


def test_missing_artifacts_give_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FAIR_INTERVENTION_ARTIFACTS", str(tmp_path))
    serve.get_bundle.cache_clear()
    serve.get_policy.cache_clear()
    with pytest.raises(RuntimeError, match="fair-intervention train"):
        serve.get_bundle()
    assert isinstance(serve.get_policy(), ThresholdPolicy)
    serve.get_bundle.cache_clear()
    serve.get_policy.cache_clear()


def test_report_writes_results_and_two_charts(tables, tmp_path):
    from fair_intervention import report

    res = report.build(
        tables,
        "Synthetic test data",
        env_config=EnvConfig(n_students=15, n_weeks=4, budget=15.0),
        train_episodes=100,
        eval_episodes=20,
        lambdas=(0.0, 10.0),
    )
    text = report.write(res, tmp_path)
    assert "Synthetic test data" in text and "Group thresholds" in text
    assert (tmp_path / "results.md").read_text() == text
    for name in ("recall_by_group.png", "lambda_tradeoff.png"):
        assert (tmp_path / "img" / name).stat().st_size > 10_000
