import numpy as np
import pytest

from dirconn.headmodel import (
    MODALITIES,
    HeadModel,
    aggregate_rois,
    average_reference,
    label_columns,
    load_head_model,
    relative_visibility,
)


class TestLabelColumns:
    vertno = (np.array([1, 4, 7, 9]), np.array([2, 3, 8]))  # 4 left and 3 right sources

    def test_left_hemisphere_labels_index_the_first_block(self):
        cols = label_columns(self.vertno, np.array([4, 9, 100]), "lh")
        np.testing.assert_array_equal(cols, [1, 3])  # 100 is not a source and is ignored

    def test_right_hemisphere_labels_are_offset_by_the_left_sources(self):
        cols = label_columns(self.vertno, np.array([3, 8]), "rh")
        np.testing.assert_array_equal(cols, [4 + 1, 4 + 2])

    def test_a_label_without_any_source_is_an_error(self):
        with pytest.raises(ValueError, match="no source"):
            label_columns(self.vertno, np.array([5, 6]), "lh")

    def test_an_unknown_hemisphere_is_an_error(self):
        with pytest.raises(ValueError, match="hemi"):
            label_columns(self.vertno, np.array([4]), "both")


class TestAggregateRois:
    def test_each_roi_is_the_mean_of_its_vertex_gains(self):
        gain = np.arange(24, dtype=float).reshape(3, 8)
        columns = [np.array([0, 1, 2]), np.array([5]), np.array([6, 7])]
        out = aggregate_rois(gain, columns)
        assert out.shape == (3, 3)
        np.testing.assert_allclose(out[:, 0], gain[:, [0, 1, 2]].mean(axis=1))
        np.testing.assert_allclose(out[:, 1], gain[:, 5])
        np.testing.assert_allclose(out[:, 2], gain[:, [6, 7]].mean(axis=1))

    def test_opposite_signed_vertices_cancel_as_they_do_in_the_cortex(self):
        # Fixed-orientation dipoles follow the surface normal, so a folded patch partly cancels.
        gain = np.array([[1.0, -1.0, 1.0, -1.0]])
        out = aggregate_rois(gain, [np.arange(4)])
        assert out[0, 0] == 0.0

    def test_an_empty_roi_is_an_error(self):
        with pytest.raises(ValueError, match="no source"):
            aggregate_rois(np.ones((2, 3)), [np.array([0]), np.array([], dtype=int)])

    def test_a_column_outside_the_gain_matrix_is_an_error(self):
        with pytest.raises(ValueError, match="outside"):
            aggregate_rois(np.ones((2, 3)), [np.array([3])])


class TestAverageReference:
    def test_every_column_sums_to_zero_over_the_channels(self):
        gain = np.random.default_rng(0).standard_normal((6, 9)) + 5.0  # a common-mode offset
        out = average_reference(gain)
        np.testing.assert_allclose(out.sum(axis=0), 0.0, atol=1e-12)

    def test_only_the_common_mode_is_removed(self):
        gain = np.random.default_rng(1).standard_normal((6, 4))
        offset = np.array([3.0, -2.0, 7.0, 1.0])
        out = average_reference(gain + offset[None, :])
        np.testing.assert_allclose(out, average_reference(gain), atol=1e-12)

    def test_it_is_idempotent_and_does_not_change_its_input(self):
        gain = np.random.default_rng(2).standard_normal((5, 3)) + 2.0
        before = gain.copy()
        once = average_reference(gain)
        np.testing.assert_array_equal(gain, before)
        np.testing.assert_allclose(average_reference(once), once, atol=1e-12)


class TestRelativeVisibility:
    def test_the_median_roi_has_visibility_one(self):
        topo = np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(relative_visibility(topo), [0.5, 1.0, 2.0])

    def test_an_invisible_roi_is_an_error_not_a_zero(self):
        topo = np.array([[1.0, 0.0, 2.0]])
        with pytest.raises(ValueError, match="invisible"):
            relative_visibility(topo)


class TestHeadModelRecord:
    @staticmethod
    def build(**changes):
        base = {
            "modality": "eeg",
            "channel_names": ("a", "b"),
            "roi_names": ("x-lh", "y-rh"),
            "roi_leadfield": np.ones((2, 2)),
            "gain": np.ones((2, 5)),
            "source_roi": np.array([0, 0, 1, -1, 1]),
            "source_flip": np.array([1, -1, 1, 0, -1]),
            "roi_centre": np.array([0, 2]),
            "scale": 2.0,
            "visibility": np.ones(2),
            "forward": None,
            "info": None,
            "provenance": {"parc": "aparc"},
        }
        base.update(changes)
        return HeadModel(**base)

    def test_a_consistent_record_is_accepted_and_read_only(self):
        model = self.build()
        assert model.n_channels == 2 and model.n_rois == 2 and model.n_sources == 5
        for array in (
            model.roi_leadfield,
            model.gain,
            model.source_roi,
            model.source_flip,
            model.visibility,
        ):
            assert not array.flags.writeable

    @pytest.mark.parametrize(
        "changes",
        [
            {"roi_leadfield": np.ones((3, 2))},
            {"roi_leadfield": np.ones((2, 3))},
            {"gain": np.ones((3, 5))},
            {"source_roi": np.array([0, 1])},
            {"source_roi": np.array([0, 0, 2, -1, 1])},
            {"roi_centre": np.array([0])},
            {"roi_centre": np.array([0, 3])},  # source 3 lies outside every region
            {"roi_centre": np.array([2, 2])},  # source 2 belongs to region 1, not region 0
            {"source_flip": np.array([1, -1, 1])},
            {"source_flip": np.array([1, -1, 2, 0, -1])},
            {
                "source_flip": np.array([1, -1, 1, 1, -1])
            },  # a source outside every region has no flip
            {"visibility": np.ones(3)},
            {"modality": "ecog"},
        ],
    )
    def test_inconsistent_records_are_refused(self, changes):
        with pytest.raises(ValueError):
            self.build(**changes)


def test_the_modalities_are_the_three_sensor_types_of_the_sample_data():
    assert MODALITIES == ("eeg", "mag", "grad")


def test_an_unknown_modality_names_the_choices_before_any_data_are_read():
    with pytest.raises(ValueError, match="eeg.*mag.*grad"):
        load_head_model("meg")


@pytest.fixture(scope="module")
def models():
    return {m: load_head_model(m) for m in MODALITIES}


@pytest.mark.sample_data
class TestSampleHeadModel:
    def test_channel_counts_exclude_the_bad_channels(self, models):
        assert {m: models[m].n_channels for m in MODALITIES} == {"eeg": 59, "mag": 102, "grad": 203}
        assert "EEG 053" not in models["eeg"].channel_names
        assert "MEG 2443" not in models["mag"].channel_names + models["grad"].channel_names

    def test_the_rois_are_the_68_dk_labels_in_mne_naming(self, models):
        names = models["eeg"].roi_names
        assert len(names) == 68 and len(set(names)) == 68
        assert "bankssts-lh" in names and "inferiortemporal-rh" in names
        assert not any(n.startswith("unknown") for n in names)
        assert models["mag"].roi_names == names == models["grad"].roi_names

    def test_the_roi_leadfield_is_the_mean_vertex_gain_computed_another_way(self, models):
        import mne

        from dirconn.datasets import require_sample

        root = require_sample()
        model = models["eeg"]
        labels = {
            lab.name: lab
            for lab in mne.read_labels_from_annot(
                "sample", parc="aparc", subjects_dir=root / "subjects", verbose="ERROR"
            )
        }
        fwd = model.forward
        n_left = len(fwd["src"][0]["vertno"])
        names = [fwd["info"]["ch_names"][i] for i in range(fwd["info"]["nchan"])]
        rows = [names.index(c) for c in model.channel_names]
        # MNE stores the gain as float32; average in float64 like the loader does
        raw = fwd["sol"]["data"][rows].astype(float)
        raw = (raw - raw.mean(axis=0, keepdims=True)) / model.scale  # EEG is average-referenced
        for roi in ("bankssts-lh", "precuneus-rh", "insula-lh"):
            restricted = labels[roi].copy().restrict(fwd["src"])
            offset = 0 if roi.endswith("lh") else n_left
            vertno = fwd["src"][0 if roi.endswith("lh") else 1]["vertno"]
            cols = offset + np.searchsorted(vertno, restricted.vertices)
            expected = raw[:, cols].mean(axis=1)
            got = model.roi_leadfield[:, model.roi_names.index(roi)]
            np.testing.assert_allclose(got, expected, rtol=1e-9)

    def test_eeg_is_average_referenced_and_meg_is_left_alone(self, models):
        # The stored EEG forward carries a large common-mode component (infinite reference): the
        # column sums over the channels are about 3.7 column norms. A recording is almost always
        # average-referenced, so the head model is too.
        eeg = models["eeg"]
        for matrix in (eeg.gain, eeg.roi_leadfield):
            np.testing.assert_allclose(
                matrix.sum(axis=0) / np.linalg.norm(matrix, axis=0), 0.0, atol=1e-9
            )
        assert eeg.provenance["reference"] == "average"
        for name in ("mag", "grad"):
            assert models[name].provenance["reference"] == "none"
            assert np.abs(models[name].gain.sum(axis=0)).max() > 1e-3  # not centred

    def test_the_forward_is_fixed_orientation_and_the_gain_matches_it(self, models):
        model = models["grad"]
        assert model.forward["source_ori"] == 1  # FIFFV_MNE_FIXED_ORI
        assert model.gain.shape == (203, model.forward["nsource"])

    def test_the_scale_makes_the_median_roi_topography_one(self, models):
        for model in models.values():
            norms = np.linalg.norm(model.roi_leadfield, axis=0)
            assert np.median(norms) == pytest.approx(1.0)
            assert model.scale > 0

    def test_visibility_is_reported_and_no_roi_is_invisible(self, models):
        # measured: max/min of the relative visibility is 12 (EEG, average-referenced), 13
        # (magnetometers) and 10 (gradiometers); the weakest ROI is at 0.22 to 0.30 of the median
        for model in models.values():
            v = model.visibility
            assert v.min() > 0.15 and v.max() < 4.0
            np.testing.assert_allclose(v, np.linalg.norm(model.roi_leadfield, axis=0))

    def test_most_sources_belong_to_a_roi_and_the_others_are_marked(self, models):
        source_roi = models["eeg"].source_roi
        assert (source_roi >= 0).sum() == 6940  # of 7498 sources
        assert set(np.unique(source_roi)) == set(range(-1, 68))

    def test_the_flips_are_mnes_label_sign_flips(self, models):
        import mne

        from dirconn.datasets import require_sample

        model = models["grad"]
        labels = {
            lab.name: lab
            for lab in mne.read_labels_from_annot(
                "sample",
                parc="aparc",
                subjects_dir=require_sample() / "subjects",
                verbose="ERROR",
            )
        }
        for roi in ("bankssts-lh", "precuneus-rh", "insula-lh"):
            expected = mne.label.label_sign_flip(labels[roi], model.forward["src"])
            columns = np.flatnonzero(model.source_roi == model.roi_names.index(roi))
            np.testing.assert_array_equal(model.source_flip[columns], expected)
        outside = model.source_roi == -1
        assert (model.source_flip[outside] == 0).all()
        assert set(np.unique(model.source_flip[~outside])) == {-1, 1}

    def test_the_centre_source_of_a_region_is_the_one_nearest_to_its_centroid(self, models):
        model = models["grad"]
        src = model.forward["src"]
        positions = np.concatenate([s["rr"][s["vertno"]] for s in src])
        for k in (0, 17, 40, 67):
            members = np.flatnonzero(model.source_roi == k)
            expected = members[
                np.argmin(
                    np.linalg.norm(positions[members] - positions[members].mean(axis=0), axis=1)
                )
            ]
            assert model.roi_centre[k] == expected
        assert all(model.source_roi[model.roi_centre[k]] == k for k in range(model.n_rois))

    def test_the_info_describes_exactly_the_channels_of_the_model(self, models):
        for model in models.values():
            assert tuple(model.info["ch_names"]) == model.channel_names
            assert model.info["bads"] == []
        eeg = models["eeg"].info
        assert [p["desc"] for p in eeg["projs"]] == ["Average EEG reference"]
        for name in ("mag", "grad"):
            assert models[name].info["projs"] == []  # the recording's own SSP is not applied

    def test_loading_twice_returns_the_cached_model(self, models):
        assert load_head_model("eeg") is models["eeg"]
