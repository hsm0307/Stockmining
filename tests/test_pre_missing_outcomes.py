"""Unknown future returns must not become successful non-crash outcomes."""
import numpy as np
import pandas as pd
import pytest

from krxlimit import analysis_pre


def _events():
    cols = ["car_60", "abvol_60", "volat_60", "range_60", "vol_trend_60", "spikes_60",
            "lowvol_60", "shadow_60", "cmf_60", "obv_60", "uptouch_60", "car_20", "abvol_20"]
    # Two distinct feature profiles yield two 32-event clusters. Non-default indices exercise
    # alignment of the return outcomes with the feature-complete event rows.
    ev = pd.DataFrame({c: np.repeat([-1.0, 1.0], 32) for c in cols}, index=np.arange(100, 164))
    ev["post_1m"] = np.repeat([0.05, 0.10], 32)
    ev["post_6m"] = ev["post_1m"]
    ev["post_3m"] = np.r_[np.tile([-0.30, -0.50, 0.20, np.nan], 8), np.full(32, np.nan)]
    return ev


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.setattr(analysis_pre, "save_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_pre, "save_fig", lambda fig, *args, **kwargs: analysis_pre.plt.close(fig))
    return lambda ev: analysis_pre._cluster(ev, k_range=(2,))


def test_cluster_crash_rate_uses_observed_outcomes(cluster):
    out = cluster(_events())
    observed_group = out.loc[out["post_1m"] == 0.05].iloc[0]
    assert observed_group["n"] == 32
    assert observed_group["3m수익률관측수"] == 24
    assert observed_group["3m수익률미관측수"] == 8
    assert observed_group["급락건수_3m(≤-30%)"] == 16
    # Includes an exact -30% return and excludes all eight unknown outcomes.
    assert observed_group["급락비율_3m(≤-30%)"] == pytest.approx(16 / 24)


def test_cluster_all_missing_outcomes_remain_unknown(cluster):
    out = cluster(_events())
    missing_group = out.loc[out["post_1m"] == 0.10].iloc[0]
    assert missing_group["n"] == 32
    assert missing_group["3m수익률관측수"] == 0
    assert missing_group["3m수익률미관측수"] == 32
    assert missing_group["급락건수_3m(≤-30%)"] == 0
    assert pd.isna(missing_group["급락비율_3m(≤-30%)"])


def test_cluster_counts_only_feature_complete_members(cluster):
    ev = _events()
    ev.loc[100, "car_60"] = np.nan  # This -30% event is not assigned to any cluster.
    out = cluster(ev)
    assert out["n"].sum() == 63
    assert out["3m수익률관측수"].sum() == 23
    assert out["급락건수_3m(≤-30%)"].sum() == 15
    assert (out["3m수익률관측수"] + out["3m수익률미관측수"]).equals(out["n"])
