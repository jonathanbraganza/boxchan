from types import SimpleNamespace

import pytest

from colour_roles import (
    COLOUR_ROLES,
    is_managed_colour_role,
)


@pytest.mark.parametrize("colour_name", list(COLOUR_ROLES.keys()))
def test_all_colour_roles_are_recognised(colour_name):
    role = SimpleNamespace(name=colour_name)

    assert is_managed_colour_role(role) is True


@pytest.mark.parametrize("colour_name", list(COLOUR_ROLES.keys()))
def test_colour_role_matching_is_case_insensitive(colour_name):
    role = SimpleNamespace(name=colour_name.swapcase())

    assert is_managed_colour_role(role) is True


def test_unrelated_role_is_not_recognised():
    role = SimpleNamespace(name="Moderator")

    assert is_managed_colour_role(role) is False


def test_role_without_name_is_not_recognised():
    role = SimpleNamespace()

    assert is_managed_colour_role(role) is False
