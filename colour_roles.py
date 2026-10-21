# Colour role selector.
from __future__ import annotations

import logging
from typing import Optional

import discord

log = logging.getLogger("colour_roles")

# Exact role names, hex colours, and display emojis
COLOUR_ROLES: dict[str, dict[str, str]] = {
    "Crimson": {"hex": "#FF3333", "emoji": "🔴"},
    "Tangerine": {"hex": "#FF8C00", "emoji": "🟠"},
    "Sunburst": {"hex": "#FFD700", "emoji": "🟡"},
    "Clover": {"hex": "#20C45A", "emoji": "🟢"},
    "Sapphire": {"hex": "#1685FF", "emoji": "🔵"},
    "Amethyst": {"hex": "#8B35D1", "emoji": "🟣"},
    "Bubblegum": {"hex": "#FF4FA3", "emoji": "🌸"},
    "Graphite": {"hex": "#7A7A7A", "emoji": "🪙"},
    "Pearl": {"hex": "#FFFFFF", "emoji": "⚪"},
    "Onyx": {"hex": "#222222", "emoji": "⚫"},
}

MANAGED_COLOUR_NAMES: frozenset[str] = frozenset(COLOUR_ROLES.keys())
MANAGED_COLOUR_NAMES_LOWER: frozenset[str] = frozenset(name.casefold() for name in MANAGED_COLOUR_NAMES)

CUSTOM_ID_SELECT: str = "boxchan:colour_roles_select"


def is_managed_colour_role(role: discord.Role) -> bool:
    """Return True if role is one of the 10 managed colour roles.

    Uses case-insensitive matching to protect against subtle casing differences.
    """
    name = getattr(role, "name", None)
    return isinstance(name, str) and name.casefold() in MANAGED_COLOUR_NAMES_LOWER


def get_member_colour_roles(member: discord.Member) -> list[discord.Role]:
    """Return any managed colour roles currently held by the member.

    Guarantees that unrelated roles (e.g. Member, Moderator, Administrator)
    are strictly ignored and never included.
    """
    roles = getattr(member, "roles", [])
    return [r for r in roles if is_managed_colour_role(r)]


async def apply_member_colour(
    member: discord.Member,
    guild: discord.Guild,
    chosen_colour: Optional[str],
) -> tuple[bool, str]:
    """Apply or remove a member's colour role with safety checks and rollback.

    Args:
        member: The Discord member whose colour is being updated.
        guild: The Discord server where the interaction occurred.
        chosen_colour: The name of the selected colour, or None/'none' to remove.

    Returns:
        (success: bool, user_message: str)
    """
    me = guild.me
    if me is None or not me.guild_permissions.manage_roles:
        return False, (
            "I cannot update your colour because I lack the **Manage Roles** permission. "
            "Please ask an administrator to grant me this permission in Server Settings."
        )

    current_colour_roles = get_member_colour_roles(member)

    # --- Scenario A: User wants to remove their colour ---
    if chosen_colour is None or chosen_colour == "none":
        if not current_colour_roles:
            return True, "You do not currently have a colour role assigned."

        for r in current_colour_roles:
            if r.position >= me.top_role.position:
                return False, (
                    f"I cannot remove the **{r.name}** role because it is positioned higher than "
                    "or equal to my role in Server Settings > Roles. Please notify an administrator."
                )

        try:
            await member.remove_roles(*current_colour_roles, reason="Colour role cleared by user")
            return True, "Removed your colour role! Your name is back to default."
        except discord.Forbidden:
            return False, "Discord prevented me from removing your role due to role hierarchy. Please notify an administrator."
        except discord.HTTPException as e:
            log.warning("Failed to remove colour role for user %s: %s", member.id, e)
            return False, "A Discord network error occurred while removing your role. Please try again later."

    # --- Scenario B: User wants to select a colour ---
    if not isinstance(chosen_colour, str) or chosen_colour.casefold() not in MANAGED_COLOUR_NAMES_LOWER:
        return False, (
            "Invalid colour selection. Please select one of the approved colours from the dropdown."
        )

    target_role = discord.utils.find(lambda r: r.name.casefold() == chosen_colour.casefold(), guild.roles)
    if target_role is None:
        return False, (
            f"The **{chosen_colour}** role does not exist on this server yet. "
            "Please ask an administrator to create it in Server Settings > Roles."
        )

    # Check hierarchy on the target role
    if target_role.position >= me.top_role.position:
        return False, (
            f"I cannot assign the **{target_role.name}** role because it is positioned higher than "
            "or equal to my role in Server Settings > Roles. Please ask an administrator to move my bot role above it."
        )

    # If the user already has this exact colour role and no other colour roles, no action needed
    if len(current_colour_roles) == 1 and current_colour_roles[0].id == target_role.id:
        return True, f"You already have the **{target_role.name}** colour!"

    # Identify existing colour roles to remove (excluding the target role if already held)
    roles_to_remove = [r for r in current_colour_roles if r.id != target_role.id]

    # Pre-check hierarchy on all roles that will be removed
    for r in roles_to_remove:
        if r.position >= me.top_role.position:
            return False, (
                f"I cannot change your colour because your current role (**{r.name}**) is positioned higher "
                "than or equal to my role in Server Settings > Roles. Please ask an administrator for help."
            )

    # 1. Remove previous colour role(s) first
    if roles_to_remove:
        try:
            await member.remove_roles(*roles_to_remove, reason="Colour role swap")
        except discord.Forbidden:
            return False, "Discord prevented me from removing your previous colour role due to permissions."
        except discord.HTTPException as e:
            log.warning("Failed to remove old colour roles for user %s: %s", member.id, e)
            return False, "A Discord network error occurred while updating your roles. Please try again later."

    # 2. Add the new colour role, with rollback if adding fails
    try:
        await member.add_roles(target_role, reason="Colour role selected by user")
        return True, f"Your colour has been updated to **{target_role.name}**!"
    except (discord.Forbidden, discord.HTTPException) as add_err:
        log.warning("Failed to add colour role %s to user %s: %s", target_role.name, member.id, add_err)

        # Rollback: attempt to restore previously removed colour roles
        if roles_to_remove:
            try:
                await member.add_roles(*roles_to_remove, reason="Rollback: restoring previous colour after failed swap")
                return False, (
                    f"Failed to assign **{target_role.name}**. "
                    f"Your previous colour has been restored."
                )
            except (discord.Forbidden, discord.HTTPException) as restore_err:
                log.error("Rollback failed for user %s: %s", member.id, restore_err)
                return False, (
                    f"Failed to assign **{target_role.name}**, and could not restore your previous colour. "
                    "Please ask an administrator for assistance."
                )

        return False, f"Failed to assign **{target_role.name}**. Please try again later."


class ColourSelect(discord.ui.Select):
    """Dropdown menu containing the colour options + remove colour option."""

    def __init__(self):
        options: list[discord.SelectOption] = []
        for name, data in COLOUR_ROLES.items():
            options.append(
                discord.SelectOption(
                    label=name,
                    value=name,
                    description=f"Hex: {data['hex']}",
                    emoji=data["emoji"],
                )
            )
        options.append(
            discord.SelectOption(
                label="Remove Colour",
                value="none",
                description="Clear your name colour",
                emoji="❌",
            )
        )
        super().__init__(
            placeholder="Select a name colour...",
            min_values=1,
            max_values=1,
            options=options,
            custom_id=CUSTOM_ID_SELECT,
        )

    async def callback(self, interaction: discord.Interaction):
        """Handle member dropdown selection safely."""
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("This menu can only be used inside a server.", ephemeral=True)
            return

        try:
            if not self.values:
                await interaction.response.send_message("Please select a colour option.", ephemeral=True)
                return

            selected_value = self.values[0]
            chosen = None if selected_value == "none" else selected_value
            success, message = await apply_member_colour(interaction.user, interaction.guild, chosen)
            await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            log.exception("Unexpected error in colour dropdown callback for user %s", interaction.user.id)
            error_msg = "An unexpected error occurred while updating your colour. Please try again."
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(error_msg, ephemeral=True)
                else:
                    await interaction.followup.send(error_msg, ephemeral=True)
            except Exception:
                pass


class ColourRolesView(discord.ui.View):
    """Persistent view holding the ColourSelect dropdown.

    Setting timeout=None ensures this view continues to function indefinitely
    and across bot restarts when registered with bot.add_view().
    """

    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(ColourSelect())
