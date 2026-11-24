"""Lesson 2.3, also try: the same image tool behind an MCP server, over stdio.

A client starts it as a subprocess: nb-gpu-tools.ipynb, or an IDE agent with
    {"command": "uv", "args": ["run", "python", "mcp_server.py"]}
MCP_FAKE=1 serves placeholder images. RUN_ID puts the assets in that run's folder.
The Endpoint token stays in this process; the client and the model never see it.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer  # mcp 2.x: FastMCP was renamed to MCPServer

from fake import PlaceholderImages
from models import SANA
from tools import Prompt, SanaEndpoint, Size, Store, image_asset, optional_setting, settings
from tracing import run_folder

server = MCPServer("media-tools")
if os.getenv("MCP_FAKE"):
    images = PlaceholderImages()
else:
    env = settings("SANA_BASE_URL", "SANA_TOKEN", "SANA_ENDPOINT_ID", ask=False)
    images = SanaEndpoint(env["SANA_BASE_URL"], env["SANA_TOKEN"], env["SANA_ENDPOINT_ID"])
LESSON = "03-gpu-tools"  # assets go to code/runs/03-gpu-tools/<RUN_ID or mcp>/
store = Store(run_folder(os.getenv("RUN_ID", "mcp"), LESSON), bucket=optional_setting("BUCKET"))


@server.tool()
async def generate_image(prompt: Prompt, seed: int = SANA.seed, size: Size = "832x480") -> dict:
    """Generate one keyframe image and return a reference to the PNG.

    Args:
        prompt: One scene, described visually. No on-screen text.
        seed: Change it to get a different image for the same prompt.
        size: WIDTHxHEIGHT in pixels. 832x480 matches the video frame.
    """
    return (await image_asset(images, store, prompt, seed, size)).model_dump()


if __name__ == "__main__":
    server.run("stdio")
