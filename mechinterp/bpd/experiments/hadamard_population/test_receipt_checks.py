"""Small, constructed receipt fixtures; no dependency on replay artifacts."""
import copy
import gzip
import json

import numpy as np
import pytest

from receipt_checks import ARMS, starts, validate


def fixture(family):
    args = dict(root=1, roots=1, cases=1, population=1, chunk=2, rounds=2, lengths=[3], archive_mib=8)
    xs, digest = starts(100300, 3, 1)
    energy = int((xs[0] * np.roll(xs[0], -1, axis=1)).sum()) ** 2
    rows = []
    for arm in ARMS[family]:
        z = dict(root=1, m=3, case=0, seed=100300, arm=arm, start_hash=digest,
                 solved=energy == 0, certificate=xs[0].tolist() if energy == 0 else None,
                 best=energy, seconds=0.0, trace=[])
        if family == "allocation":
            z.update(candidate_scores=0, offers=0, transfers=0, allocation=[0], rounds=0)
        else:
            z.update(candidate_units=0, archive_candidate_units=0, search_candidate_scores=0,
                     solution_kind="native" if energy == 0 else None)
        if family == "near":
            z.update(injections=[], solution_injection=0 if energy == 0 else None)
        rows.append(z)
    return dict(args=args, rows=rows)


def write(tmp_path, data, compressed=False):
    path = tmp_path / ("receipt.json.gz" if compressed else "receipt.json")
    text = json.dumps(data)
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("family", ARMS)
@pytest.mark.parametrize("compressed", [False, True])
def test_full_small_roster_accepts_json_and_gzip(tmp_path, family, compressed):
    result = validate(write(tmp_path, fixture(family), compressed), family)
    assert result["rows"] == len(ARMS[family])
    assert result["paired_populations"] == 1
    assert result["status"] == "finite_integrity_pass"


def test_missing_arm_is_rejected(tmp_path):
    data = fixture("exact")
    data["rows"].pop()
    with pytest.raises(ValueError, match="full declared roster"):
        validate(write(tmp_path, data), "exact")


def test_duplicate_arm_is_rejected(tmp_path):
    data = fixture("exact")
    data["rows"][-1] = copy.deepcopy(data["rows"][0])
    with pytest.raises(ValueError, match="duplicates a roster"):
        validate(write(tmp_path, data), "exact")


def test_wrong_start_hash_is_rejected(tmp_path):
    data = fixture("allocation")
    data["rows"][0]["start_hash"] = "0" * 64
    with pytest.raises(ValueError, match="regenerated starts"):
        validate(write(tmp_path, data), "allocation")


def test_negative_subcost_cannot_hide_in_a_balanced_total(tmp_path):
    data = fixture("exact")
    data["rows"][0].update(archive_candidate_units=-12, search_candidate_scores=12)
    with pytest.raises(ValueError, match="archive work must be an integer"):
        validate(write(tmp_path, data), "exact")


def test_trace_work_cannot_go_backwards(tmp_path):
    data = fixture("exact")
    z = data["rows"][0]
    z.update(candidate_units=24, search_candidate_scores=24,
             trace=[[0, z["best"], z["best"], 24, 0], [0, z["best"], z["best"], 12, 0]])
    with pytest.raises(ValueError, match="trace work is not monotone"):
        validate(write(tmp_path, data), "exact")


def test_injection_cannot_reference_a_future_generation(tmp_path):
    data = fixture("near")
    z = next(row for row in data["rows"] if row["arm"] == "near64")
    xs, _ = starts(100300, 3, 1)
    z.update(candidate_units=12, archive_candidate_units=12)
    z["injections"] = [dict(generation=1, slot=0, energy=0, replaced_best=16,
                             work=12, sequences=xs[0].tolist(),
                             sources=[[0, 0, k, 2] for k in range(4)])]
    with pytest.raises(ValueError, match="future generation"):
        validate(write(tmp_path, data), "near")


def test_block_zero_source_row_is_reconstructed(tmp_path):
    data = fixture("near")
    z = next(row for row in data["rows"] if row["arm"] == "near64")
    xs, _ = starts(100300, 3, 1)
    child = xs[0].tolist()
    child[0][0] *= -1
    z.update(candidate_units=12, archive_candidate_units=12)
    z["injections"] = [dict(generation=1, slot=0, energy=0, replaced_best=16,
                             work=12, sequences=child,
                             sources=[[0, 0, k, 0] for k in range(4)])]
    with pytest.raises(ValueError, match="genesis source row mismatch"):
        validate(write(tmp_path, data), "near")


def test_serialized_injection_count_is_checked_per_row(tmp_path):
    data = fixture("near")
    data["rows"][0]["injection_count"] = 1
    with pytest.raises(ValueError, match="count differs from serialized events"):
        validate(write(tmp_path, data), "near")
