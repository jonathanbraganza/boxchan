import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

import botimport  # noqa: F401  (must be imported before bot)
import bot
from colour_roles import (
    COLOUR_ROLES,
    apply_member_colour,
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


def test_none_removal_selection_is_case_insensitive():
    async def run_test():
        guild = SimpleNamespace(
            me=SimpleNamespace(
                guild_permissions=SimpleNamespace(manage_roles=True),
                top_role=SimpleNamespace(position=10),
            )
        )
        role = SimpleNamespace(id=11, name="Crimson", position=1)
        member = SimpleNamespace(
            id=42,
            roles=[role],
            remove_roles=mock.AsyncMock(),
            add_roles=mock.AsyncMock(),
        )

        success, message = await apply_member_colour(member, guild, "NONE")

        assert success is True
        assert "Removed your colour role" in message
        member.remove_roles.assert_awaited_once()
        assert member.remove_roles.await_args.args == (role,)
        assert member.remove_roles.await_args.kwargs["reason"] == "Colour role cleared by user"

    asyncio.run(run_test())


def test_can_post_colourpanel_allows_big_bots_role():
    role = SimpleNamespace(name="Big bots")
    author = mock.Mock(spec=bot.discord.Member)
    author.guild_permissions = SimpleNamespace(manage_roles=False, administrator=False)
    author.roles = [role]
    ctx = SimpleNamespace(guild=object(), author=author)

    assert bot.can_post_colourpanel(ctx) is True


def test_can_post_colourpanel_rejects_regular_member():
    author = mock.Mock(spec=bot.discord.Member)
    author.guild_permissions = SimpleNamespace(manage_roles=False, administrator=False)
    author.roles = [SimpleNamespace(name="Member")]
    ctx = SimpleNamespace(guild=object(), author=author)

    assert bot.can_post_colourpanel(ctx) is False


def test_colourpanel_error_mentions_big_bots_permission_exception():
    async def run_test():
        ctx = SimpleNamespace(interaction=None, send=mock.AsyncMock())

        await bot.colourpanel_error(ctx, bot.commands.CheckFailure("x"))

        ctx.send.assert_awaited_once()
        message = ctx.send.await_args.args[0]
        assert "Big bots" in message
        assert "Manage Roles" in message
        assert "Administrator" in message

    asyncio.run(run_test())
