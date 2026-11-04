from pathlib import Path

import mne
import pytest

from dirconn.datasets import SAMPLE_FILES, SampleDataMissingError, require_sample


def make_tree(root: Path, skip: tuple[str, ...] = ()) -> Path:
    """Create an (empty-file) MNE-sample-data tree under ``root``."""
    sample = root / "MNE-sample-data"
    for name in SAMPLE_FILES:
        if name in skip:
            continue
        target = sample / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    sample.mkdir(parents=True, exist_ok=True)
    return sample


class SpyDataPath:
    """Stands in for ``mne.datasets.sample.data_path``; never touches the network."""

    def __init__(self, root: Path, *, provides: bool = True):
        self.root, self.provides, self.calls = root, provides, []

    def __call__(self, path=None, **kwargs):
        self.calls.append({"path": path, **kwargs})
        if kwargs.get("download") and self.provides:
            make_tree(self.root)
        return self.root / "MNE-sample-data"


@pytest.fixture
def spy(tmp_path, monkeypatch):
    def install(*, tree=True, skip=(), provides=True):
        if tree:
            make_tree(tmp_path, skip)
        fake = SpyDataPath(tmp_path, provides=provides)
        monkeypatch.setattr(mne.datasets.sample, "data_path", fake)
        return fake

    return install


def test_a_complete_local_copy_is_returned_and_nothing_is_downloaded(tmp_path, spy):
    fake = spy()
    assert require_sample() == tmp_path / "MNE-sample-data"
    assert len(fake.calls) == 1 and fake.calls[0]["download"] is False


def test_the_global_mne_configuration_is_never_changed(tmp_path, spy):
    # data_path(update_path=True) rewrites MNE's config file; a check must not do that.
    fake = spy()
    require_sample()
    assert fake.calls[0]["update_path"] is False


def test_missing_data_stop_with_instructions_and_do_not_download(tmp_path, spy):
    fake = spy(tree=False)
    with pytest.raises(SampleDataMissingError) as info:
        require_sample()
    text = str(info.value)
    assert "1.5 GB" in text and "MNE_DATA" in text and "download=True" in text
    assert "data_path(download=True)" in text
    assert all(call["download"] is False for call in fake.calls)


def test_the_error_is_a_file_not_found_error():
    assert issubclass(SampleDataMissingError, FileNotFoundError)


def test_a_partial_copy_names_the_missing_files(tmp_path, spy):
    gone = SAMPLE_FILES[0], SAMPLE_FILES[-1]
    fake = spy(skip=gone)
    with pytest.raises(SampleDataMissingError) as info:
        require_sample()
    assert all(name in str(info.value) for name in gone)
    assert all(call["download"] is False for call in fake.calls)


def test_a_download_happens_only_when_asked_for_and_only_once(tmp_path, spy):
    fake = spy(tree=False)
    assert require_sample(download=True) == tmp_path / "MNE-sample-data"
    downloads = [c for c in fake.calls if c["download"]]
    assert len(downloads) == 1 and downloads[0]["update_path"] is False


def test_asking_for_a_download_when_the_data_are_there_does_not_download(tmp_path, spy):
    fake = spy()
    require_sample(download=True)
    assert all(call["download"] is False for call in fake.calls)


def test_a_download_that_does_not_supply_the_files_is_an_error(tmp_path, spy):
    spy(tree=False, provides=False)
    with pytest.raises(SampleDataMissingError, match="still missing"):
        require_sample(download=True)


def test_an_explicit_path_is_passed_on(tmp_path, spy):
    fake = spy()
    require_sample(path=tmp_path)
    assert fake.calls[0]["path"] == tmp_path


def test_a_path_that_does_not_exist_is_reported_before_mne_is_asked(tmp_path, spy):
    fake = spy()
    with pytest.raises(SampleDataMissingError, match="does not exist"):
        require_sample(path=tmp_path / "nowhere")
    assert fake.calls == []


@pytest.mark.parametrize("download", [1, "yes", None])
def test_download_must_be_a_bool(download):
    with pytest.raises(TypeError, match="download"):
        require_sample(download=download)  # type: ignore[arg-type]


def test_real_mne_config_is_not_modified_by_a_failed_check(tmp_path):
    # MNE writes its config when the dataset folder exists, even if it is empty.
    (tmp_path / "MNE-sample-data").mkdir()
    key = "MNE_DATASETS_SAMPLE_PATH"
    before = mne.get_config(key)
    try:
        with pytest.raises(SampleDataMissingError):
            require_sample(path=tmp_path)
        assert mne.get_config(key) == before
    finally:
        # if the guard ever fails it must not leave the user's MNE configuration pointing at a
        # temporary directory
        mne.set_config(key, before, set_env=False)


@pytest.mark.sample_data
def test_the_local_sample_dataset_has_every_file_the_benchmark_needs():
    root = require_sample()
    assert all((root / name).is_file() for name in SAMPLE_FILES)
