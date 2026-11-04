import numpy as np
import pytest

from dirconn.rng import RNGController


def draws(generators, n=3):
    return [g.random(n) for g in generators]


@pytest.mark.parametrize("seed", [None, -1, 1.5, "7", True])
def test_the_seed_is_required_and_must_be_a_non_negative_integer(seed):
    with pytest.raises((TypeError, ValueError)):
        RNGController(seed)


def test_the_seed_is_not_optional():
    with pytest.raises(TypeError):
        RNGController()  # type: ignore[call-arg]


def test_numpy_integers_are_accepted_and_equal_to_python_integers():
    a = draws(RNGController(np.int64(5)).spawn(3))
    b = draws(RNGController(5).spawn(3))
    np.testing.assert_array_equal(a, b)


def test_the_same_seed_gives_the_same_generators():
    np.testing.assert_array_equal(
        draws(RNGController(1).spawn(4)), draws(RNGController(1).spawn(4))
    )


def test_different_seeds_give_different_generators():
    assert not np.array_equal(draws(RNGController(1).spawn(4)), draws(RNGController(2).spawn(4)))


def test_the_children_are_distinct_streams():
    first = [row[0] for row in draws(RNGController(0).spawn(50))]
    assert len(set(first)) == 50


def test_spawning_again_returns_the_same_streams():
    # Stateless on purpose: a second call must not silently move on to new children.
    controller = RNGController(3)
    np.testing.assert_array_equal(draws(controller.spawn(4)), draws(controller.spawn(4)))


def test_asking_for_more_children_extends_the_same_sequence():
    # Raising n_surrogates from 100 to 1000 keeps the first 100 surrogates unchanged.
    controller = RNGController(11)
    np.testing.assert_array_equal(draws(controller.spawn(5)), draws(controller.spawn(20))[:5])


def test_a_single_child_equals_the_same_entry_of_spawn():
    # Workers build only the generators for their own share and still agree with a serial run.
    controller = RNGController(9)
    everything = draws(controller.spawn(12))
    for index in (0, 5, 11):
        np.testing.assert_array_equal(controller.child(index).random(3), everything[index])


def test_the_result_does_not_depend_on_how_the_work_is_split():
    controller = RNGController(21)
    serial = draws(controller.spawn(10))
    split = draws([controller.child(i) for i in range(0, 10, 2)]) + draws(
        [controller.child(i) for i in range(1, 10, 2)]
    )
    order = list(range(0, 10, 2)) + list(range(1, 10, 2))
    for position, index in enumerate(order):
        np.testing.assert_array_equal(split[position], serial[index])


@pytest.mark.parametrize("n", [0, -1, 2.5])
def test_the_number_of_children_must_be_a_positive_integer(n):
    with pytest.raises((TypeError, ValueError)):
        RNGController(0).spawn(n)


@pytest.mark.parametrize("index", [-1, 1.5, True])
def test_a_child_index_must_be_a_non_negative_integer(index):
    with pytest.raises((TypeError, ValueError)):
        RNGController(0).child(index)


def test_the_seed_is_kept_for_the_provenance():
    assert RNGController(42).seed == 42
