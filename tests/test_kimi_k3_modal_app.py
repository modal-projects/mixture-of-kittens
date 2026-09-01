"""CPU-only contracts for the Modal app now that it is spread across six files.

`modal run modal_app.py::verify` resolves ``verify`` as an attribute of
``modal_app``, not as a function of the app it registers. So splitting the app
up introduced a way to be half-right: import the five modules and every gate is
registered and reachable through the API, while every documented command line
stops working. These hold the two together.

Read from the source rather than by importing, because importing needs
``modal`` and this file has to hold on a machine that has no reason to carry it.
"""

from __future__ import annotations

import pytest

from . import modal_sources


def test_every_registered_name_is_reachable_by_name() -> None:
    """Whatever the app registers, `modal run modal_app.py::<name>` finds."""
    registered = modal_sources.registered()
    assert registered, "the app registers nothing"

    absent = sorted(set(registered) - modal_sources.entry_names())
    assert absent == [], (
        f"registered but not re-exported by modal_app.py: {absent}"
    )


def test_the_entry_module_imports_every_other_one() -> None:
    """A module nobody imports registers nothing, and fails silently doing it."""
    entry = modal_sources.ENTRY
    others = {
        path.stem for path in modal_sources.files() if path.name != entry
    }
    assert others, "the app was not split"

    unreached = sorted(others - modal_sources.entry_imports())
    assert unreached == [], f"never imported by {entry}: {unreached}"


@pytest.mark.parametrize(
    "command",
    ["gpu_info", "bench", "verify", "engine_probe", "compare"],
)
def test_the_documented_commands_resolve(command: str) -> None:
    """The usage block is a list of things that have to work, not a comment."""
    assert command in modal_sources.registered(), command
    assert command in modal_sources.entry_names(), command
    assert f"modal_app.py::{command}" in modal_sources.read()


def test_every_orchestration_module_travels_into_the_image() -> None:
    """A remote function is deserialized against its own module, not the entry.

    So an image carrying only `modal_app.py` would import five modules that are
    not there. The allowlist is what puts them there, and it has to name all of
    them.
    """
    source = modal_sources.read()
    listed = source.split("ORCHESTRATION_FILES = (", 1)[1].split(")", 1)[0]
    for path in modal_sources.files():
        assert f'"{path.name}"' in listed, path.name


def test_every_image_the_app_builds_carries_them() -> None:
    """Naming the allowlist is half of it; every builder has to use it.

    Modal mounts the module `modal run` was pointed at on its own, so a
    single-file app needed no allowlist at all -- and after the split the
    comparison images still had none. `compare_vllm` reached its container and
    failed to import `modal_images`, which is a twenty-minute image build and two
    eight-GPU reservations to discover. Both builders return an image, so both
    are required to add them.
    """
    builders = modal_sources.image_builders()
    assert sorted(builders) == ["build_image", "framework_comparison_image"], (
        sorted(builders)
    )
    for name, body in builders.items():
        assert "ORCHESTRATION_FILES" in body, name
