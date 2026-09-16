import pytest

from auditkit.registry import (
    Registry, ObjectSpec,
    SCENARIOS, ADAPTERS, MODELS, METRICS, ANNOTATORS, EVALUATORS,
)
from auditkit.errors import RegistryError


def test_register_and_get_by_name():
    reg = Registry("thing")

    @reg.register("widget")
    class Widget:
        pass

    assert reg.get("widget") is Widget
    assert "widget" in reg.names()


def test_get_unknown_name_raises():
    reg = Registry("thing")
    with pytest.raises(RegistryError):
        reg.get("nope")


def test_duplicate_registration_raises():
    reg = Registry("thing")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(RegistryError):
        @reg.register("dup")
        class B:
            pass


def test_create_from_objectspec_passes_args():
    reg = Registry("thing")

    @reg.register("point")
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    p = reg.create(ObjectSpec(class_name="point", args={"x": 1, "y": 2}))
    assert (p.x, p.y) == (1, 2)


def test_names_sorted_and_isolated_per_registry():
    reg = Registry("thing")

    @reg.register("b")
    class B:
        pass

    @reg.register("a")
    class A:
        pass

    assert reg.names() == ["a", "b"]


def test_global_registries_are_distinct_instances():
    regs = [SCENARIOS, ADAPTERS, MODELS, METRICS, ANNOTATORS, EVALUATORS]
    assert len({id(r) for r in regs}) == 6
    assert SCENARIOS.kind == "scenario"
    assert METRICS.kind == "metric"
