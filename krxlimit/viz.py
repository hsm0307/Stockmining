"""그래프·표 저장 공통 함수 (한글 폰트 설정 포함)."""
from __future__ import annotations

import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402


def setup_korean_font() -> None:
    try:
        import koreanize_matplotlib  # noqa: F401  (pip install koreanize-matplotlib)
        return
    except ImportError:
        pass
    font = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

UP_COLOR = "#d62728"    # 한국 관례: 상승 빨강
DOWN_COLOR = "#1f77b4"  # 하락 파랑
NEUTRAL = "#7f7f7f"


def date_axis(ax) -> None:
    """날짜 축 눈금이 겹치지 않도록 간결한 형식으로."""
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def save_fig(fig, name: str) -> str:
    config.ensure_dirs()
    path = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> str:
    """엑셀에서 한글이 깨지지 않도록 utf-8-sig 로 저장."""
    config.ensure_dirs()
    path = config.TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return str(path)


def read_table(name: str) -> pd.DataFrame | None:
    path = config.TABLE_DIR / f"{name}.csv"
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str}) if path.exists() else None  # 종목코드 앞자리 0 유지
