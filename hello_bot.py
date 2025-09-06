import os
import discord
from discord.ext import commands

# Discord.py 2.0 requires explicit intents; enable message content for commands
intents = discord.Intents.default()
intents.message_content = True

# Use when_mentioned so the bot only reacts when pinged
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

@bot.event
async def on_message(message: discord.Message) -> None:
    """Respond with "Hello, world!" only when pinged with that phrase."""
    if message.author.bot:
        return
    if bot.user in message.mentions:
        content = message.content
        for mention in message.mentions:
            content = content.replace(mention.mention, '')
        if content.strip().lower() == 'hello world':
            await message.channel.send('Hello, world!')
    # No additional command processing; the bot only reacts to mentions

if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if token is None:
        raise ValueError('DISCORD_TOKEN environment variable not set')
    bot.run(token)
