"""EuroSCORE II pulmonary hypertension cut-offs (Nashef et al. 2012).

Published definition: moderate 31–55 mmHg, severe > 55 mmHg — a PSAP of
exactly 55 mmHg is moderate, not severe.
"""

import math

import numpy as np
import pytest

from euroscore import COEF, _pap_category, euroscore_from_inputs


@pytest.mark.parametrize(
    "psap, expected",
    [
        (np.nan, "normal"),
        (20, "normal"),
        (30, "normal"),
        (31, "31_55"),
        (40, "31_55"),
        (55, "31_55"),
        (55.5, "ge_55"),
        (56, "ge_55"),
        (80, "ge_55"),
    ],
)
def test_pap_category_boundaries(psap, expected):
    assert _pap_category(float(psap)) == expected


_BASE = {
    "Age (years)": 65,
    "Sex": "M",
    "Preoperative NYHA": "II",
    "Surgery": "CABG",
    "Surgical Priority": "Elective",
    "Pré-LVEF, %": 60,
    "Cr clearance, ml/min *": 90,
}


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def test_psap_55_gets_moderate_coefficient():
    p_none = euroscore_from_inputs({**_BASE, "PSAP": None})
    p_55 = euroscore_from_inputs({**_BASE, "PSAP": 55})
    p_56 = euroscore_from_inputs({**_BASE, "PSAP": 56})
    assert _logit(p_55) - _logit(p_none) == pytest.approx(COEF["pap_31_55"], abs=1e-9)
    assert _logit(p_56) - _logit(p_none) == pytest.approx(COEF["pap_ge_55"], abs=1e-9)
