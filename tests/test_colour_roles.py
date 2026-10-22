from types import SimpleNamespace
import unittest
from unittest import mock

import botimport  # noqa: F401  (must be imported before bot)
import bot
from colour_roles import (
    COLOUR_ROLES,
    ColourSelect,
    apply_member_colour,
    is_managed_colour_role,
)


class ColourRolesTest(unittest.IsolatedAsyncioTestCase):
    def test_colour_names_and_hex_values_match_configured_roles(self):
        self.assertEqual(
            {name: data["hex"] for name, data in COLOUR_ROLES.items()},
            {
                "Sharingan Red": "#FF3333",
                "Naruto Orange": "#FF8C00",
                "Thunder Breathing": "#FFD700",
                "Deku Green": "#20C45A",
                "Limitless Blue": "#1685FF",
                "Hollow Purple": "#8B35D1",
                "Sakura Blossom": "#FF4FA3",
                "Titan Ash": "#7A7A7A",
                "Hashira White": "#FFFFFF",
                "Akatsuki Black": "#222222",
            },
        )

    def test_dropdown_uses_new_names_and_preserves_remove_option(self):
        options = ColourSelect().options

        self.assertEqual(
            [(option.label, option.value) for option in options],
            [(name, name) for name in COLOUR_ROLES] + [("Remove Colour", "none")],
        )

    def test_all_colour_roles_are_recognised(self):
        for colour_name in COLOUR_ROLES:
            with self.subTest(colour_name=colour_name):
                role = SimpleNamespace(name=colour_name)

                self.assertTrue(is_managed_colour_role(role))

    def test_colour_role_matching_is_case_insensitive(self):
        for colour_name in COLOUR_ROLES:
            with self.subTest(colour_name=colour_name):
                role = SimpleNamespace(name=colour_name.swapcase())

                self.assertTrue(is_managed_colour_role(role))

    def test_unrelated_role_is_not_recognised(self):
        role = SimpleNamespace(name="Moderator")

        self.assertFalse(is_managed_colour_role(role))

    def test_role_without_name_is_not_recognised(self):
        role = SimpleNamespace()

        self.assertFalse(is_managed_colour_role(role))

    async def test_none_removal_selection_is_case_insensitive(self):
        guild = SimpleNamespace(
            me=SimpleNamespace(
                guild_permissions=SimpleNamespace(manage_roles=True),
                top_role=SimpleNamespace(position=10),
            )
        )
        role = SimpleNamespace(id=11, name="Sharingan Red", position=1)
        member = SimpleNamespace(
            id=42,
            roles=[role],
            remove_roles=mock.AsyncMock(),
            add_roles=mock.AsyncMock(),
        )

        success, message = await apply_member_colour(member, guild, "NONE")

        self.assertTrue(success)
        self.assertIn("Removed your colour role", message)
        member.remove_roles.assert_awaited_once()
        self.assertEqual(member.remove_roles.await_args.args, (role,))
        self.assertEqual(
            member.remove_roles.await_args.kwargs["reason"],
            "Colour role cleared by user",
        )

    def test_can_post_colourpanel_allows_big_bots_role(self):
        role = SimpleNamespace(name="Big bots")
        author = mock.Mock(spec=bot.discord.Member)
        author.guild_permissions = SimpleNamespace(manage_roles=False, administrator=False)
        author.roles = [role]
        ctx = SimpleNamespace(guild=object(), author=author)

        self.assertTrue(bot.can_post_colourpanel(ctx))

    def test_can_post_colourpanel_rejects_regular_member(self):
        author = mock.Mock(spec=bot.discord.Member)
        author.guild_permissions = SimpleNamespace(manage_roles=False, administrator=False)
        author.roles = [SimpleNamespace(name="Member")]
        ctx = SimpleNamespace(guild=object(), author=author)

        self.assertFalse(bot.can_post_colourpanel(ctx))

    async def test_colourpanel_error_mentions_big_bots_permission_exception(self):
        ctx = SimpleNamespace(interaction=None, send=mock.AsyncMock())

        await bot.colourpanel_error(ctx, bot.commands.CheckFailure("x"))

        ctx.send.assert_awaited_once()
        message = ctx.send.await_args.args[0]
        self.assertIn("Big bots", message)
        self.assertIn("Manage Roles", message)
        self.assertIn("Administrator", message)
