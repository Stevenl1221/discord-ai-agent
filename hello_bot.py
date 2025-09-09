import os
import asyncio
import logging
from dotenv import load_dotenv
import discord
from discord import app_commands


logging.basicConfig(level=logging.INFO)

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")

intents = discord.Intents.default()
intents.message_content = True

class PingClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            await self.tree.sync(guild=guild)
            logging.info("Synced commands to guild %s", GUILD_ID)
        else:
            await self.tree.sync()
            logging.info("Synced global commands (may take up to 1 hour)")


client = PingClient(intents=intents)


@client.tree.command(name="ping", description="Simple health check")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong")


def main():
    if not TOKEN:
        raise SystemExit("DISCORD_TOKEN not set in .env")
    client.run(TOKEN)


if __name__ == "__main__":
    main()

