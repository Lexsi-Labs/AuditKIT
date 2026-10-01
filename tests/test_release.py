import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release import (
    Presence,
    ReleaseAction,
    ReleaseError,
    ReleaseState,
    classify_release_state,
    current_version,
    decide_action,
    next_patch,
    release_version,
    parse_version,
    set_version,
    validate_distributions,
    validate_requested_version,
)


def _version_tree(root: Path, version: str = "1.0.0") -> Path:
    (root / "src/auditkit").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (root / "src/auditkit/__init__.py").write_text(f'__version__ = "{version}"\n')
    (root / "CITATION.cff").write_text(f"version: {version}\n")
    return root


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.1.0", "0.1.1"), ("1.9.99", "1.9.100"), ("10.0.8", "10.0.9")],
)
def test_next_patch_increments_only_patch(current, expected):
    assert next_patch(current) == expected


def test_successive_merges_advance_patch_releases():
    latest = "0.1.0"
    for expected in ("0.1.1", "0.1.2", "0.1.3"):
        requested = next_patch(latest)
        assert requested == expected
        decision = decide_action(trigger="normal", version=requested, state=ReleaseState.READY)
        assert decision.action is ReleaseAction.RELEASE
        assert decision.version == expected
        latest = requested


@pytest.mark.parametrize(
    "invalid",
    ["v1.2.3", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "1.2.3rc1", "1.2.3+meta", ""],
)
def test_plain_semver_rejects_invalid_versions(invalid):
    with pytest.raises(ReleaseError):
        parse_version(invalid)


def test_initial_version_is_preserved_for_bootstrap(tmp_path):
    root = _version_tree(tmp_path, "0.1.0")
    state = classify_release_state(Presence(False, False, False))
    decision = decide_action(trigger="dispatch", version=current_version(root), state=state)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "0.1.0"
    assert decision.state is ReleaseState.READY


def test_set_version_updates_every_authoritative_location(tmp_path):
    root = _version_tree(tmp_path)
    set_version("1.0.1", root)
    assert current_version(root) == "1.0.1"
    assert "1.0.1" in (root / "pyproject.toml").read_text()
    assert "1.0.1" in (root / "src/auditkit/__init__.py").read_text()
    assert "1.0.1" in (root / "CITATION.cff").read_text()


def test_set_version_preserves_surrounding_lines(tmp_path):
    root = _version_tree(tmp_path)
    originals = {}
    for relative in ("pyproject.toml", "src/auditkit/__init__.py", "CITATION.cff"):
        path = root / relative
        originals[path] = f"before\n\n{path.read_text()}\nafter\n"
        path.write_text(originals[path])
    set_version("1.0.1", root)
    for path, original in originals.items():
        assert path.read_text() == original.replace("1.0.0", "1.0.1")

def test_requested_version_must_match_all_package_metadata(tmp_path):
    root = _version_tree(tmp_path)
    with pytest.raises(ReleaseError, match="does not match"):
        validate_requested_version("1.0.1", root)
    (root / "CITATION.cff").write_text("version: 1.0.1\n")
    with pytest.raises(ReleaseError, match="disagree"):
        current_version(root)


@pytest.mark.parametrize(
    ("presence", "expected"),
    [
        (Presence(False, False, False), ReleaseState.READY),
        (Presence(True, False, False), ReleaseState.RESUMABLE),
        (Presence(True, True, False), ReleaseState.RESUMABLE),
        (Presence(True, True, True), ReleaseState.COMPLETE),
        (Presence(False, True, False), ReleaseState.CONFLICT),
        (Presence(False, False, True), ReleaseState.CONFLICT),
        (Presence(True, False, True), ReleaseState.CONFLICT),
    ],
)
def test_release_completeness_states(presence, expected):
    assert classify_release_state(presence) is expected


def test_mismatched_or_lightweight_tag_is_conflicting():
    presence = Presence(True, False, False)
    assert classify_release_state(presence, tag_is_annotated=False) is ReleaseState.CONFLICT
    assert classify_release_state(presence, tag_matches_source=False) is ReleaseState.CONFLICT


def test_normal_merge_bootstraps_incomplete_current_version():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.READY)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "1.0.0"


def test_normal_merge_resumes_partial_current_release():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.RESUMABLE)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "1.0.0"


def test_complete_current_release_is_a_noop():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.COMPLETE)
    assert decision.action is ReleaseAction.NOOP
    assert decision.version == "1.0.0"


def test_conflicting_state_fails_decision():
    with pytest.raises(ReleaseError, match="manual repair"):
        decide_action(trigger="dispatch", version="1.0.0", state=ReleaseState.CONFLICT)


def _write_distributions(root: Path, version: str) -> list[Path]:
    metadata = f"Metadata-Version: 2.1\nName: auditkit\nVersion: {version}\n\n"
    wheel = root / f"auditkit-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"auditkit-{version}.dist-info/METADATA", metadata)
    sdist = root / f"auditkit-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = metadata.encode()
        info = tarfile.TarInfo(f"auditkit-{version}/PKG-INFO")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        nested = tarfile.TarInfo(f"auditkit-{version}/src/auditkit.egg-info/PKG-INFO")
        nested.size = len(payload)
        archive.addfile(nested, io.BytesIO(payload))
    return [wheel, sdist]


def test_distribution_metadata_must_match_requested_version(tmp_path):
    validate_distributions("1.0.0", _write_distributions(tmp_path, "1.0.0"))
    with pytest.raises(ReleaseError, match="expected exact version"):
        validate_distributions("1.0.1", list(tmp_path.iterdir()))


@pytest.mark.parametrize("current,latest,expected", [
    ("1.2.0", "1.0.0", "1.2.0"),   # a deliberate bump in the code is released as is (not 1.0.1)
    ("2.0.0", "1.9.9", "2.0.0"),
    ("1.2.0", "1.2.0", "1.2.1"),   # the code isn't ahead: next patch, as before
    ("1.2.0", "1.2.3", "1.2.4"),
    ("1.0.0", "1.2.0", "1.2.1"),   # an older code version never moves the release backwards
])
def test_release_version_honours_a_bump_in_the_code(current, latest, expected):
    assert release_version(current, latest) == expected


def test_merges_after_the_bump_advance_patch_releases():
    latest = "1.0.0"
    released = []
    for _ in range(3):
        latest = release_version("1.2.0", latest)
        released.append(latest)
    assert released == ["1.2.0", "1.2.1", "1.2.2"]


def test_the_real_init_layout_is_read_and_updated(tmp_path):
    # since 1.2.0, __init__.py reads installed metadata and keeps the literal as an indented fallback
    root = _version_tree(tmp_path, "1.2.0")
    init = root / "src/auditkit/__init__.py"
    init.write_text(
        "from importlib import metadata as _metadata\n\n"
        "try:\n"
        '    __version__ = _metadata.version("auditkit")\n'
        "except _metadata.PackageNotFoundError:\n"
        '    __version__ = "1.2.0"\n'
    )
    assert current_version(root) == "1.2.0"
    set_version("1.2.1", root)
    assert current_version(root) == "1.2.1"
    text = init.read_text()
    assert '    __version__ = "1.2.1"\n' in text and '_metadata.version("auditkit")' in text
