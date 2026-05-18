# this module utilizes the user-defined MCP servers configs to create MCP clients
import json
import os
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self):
        self.cached_tools = None  # cache to store fetched tools
        self.mcp_client = None

    async def initialize(self):
        mcp_config_path = Path(__file__).parent / "mcp-config.json"
        servers = await self._get_servers_from_config(mcp_config_path)
        client = await self._create_multi_server_mcp_client(servers=servers)
        self.mcp_client = client
        await self.fetch_mcp_tools()
        return self

    async def _create_multi_server_mcp_client(
        self, servers: dict, tool_name_prefix: bool = True
    ):
        """Create the MultiServerMCP client instance."""
        if not servers:
            logger.warning(
                "[Technical Agent MCPClient] No servers available to create MCP client"
            )
            raise ValueError(
                "[Technical Agent MCPClient] No servers available to create MCP client"
            )
        try:
            client = MultiServerMCPClient(
                connections=servers, tool_name_prefix=tool_name_prefix
            )
            logger.info("[Technical Agent MCPClient] MCP Client successfully created")
            return client
        except Exception as e:
            logger.error(f"[Technical Agent MCPClient] Error creating MCP Client: {e}")
            raise ValueError(
                f"[Technical Agent MCPClient] Error creating MCP Client: {e}"
            )

    async def _get_servers_from_config(self, mcp_config_path) -> dict:
        """Read and return the MCP server configuration."""
        try:
            # Load and parse JSON
            with open(mcp_config_path) as f:
                mcp_config = json.load(f)
            # Get servers dict (returns {} if 'servers' key missing)
            servers_dict = mcp_config.get("servers", {})
            final_dict = {}

            if not servers_dict:
                logger.warning(
                    "[Technical Agent MCPClient] No servers configured in mcp-config.json"
                )
                raise ValueError(
                    "[Technical Agent MCPClient] No servers configured in mcp-config.json"
                )

            # Substitute API keys
            for server_name, server_config in servers_dict.items():
                url = server_config.get("url", "")
                if "API_KEY" in url:
                    env_key_name = f"{server_name.upper()}_API_KEY_TA"
                    api_key = os.getenv(env_key_name)
                    if not api_key:
                        logger.error(
                            f"Missing API key for '{server_name}': set {env_key_name} env var"
                        )
                        raise ValueError(
                            f"Environment variable not set: {env_key_name}"
                        )
                    else:
                        server_config["url"] = url.format(API_KEY=api_key)
                        final_dict[server_name] = server_config
                    # logger.info(f"API key substituted for server: {server_name}")
            logger.info(f"MCP servers: {len(final_dict)}")
            return final_dict

        except FileNotFoundError as e:
            logger.error(f"[Technical Agent MCPClient] Config file missing: {e}")
            raise ValueError(f"[Technical Agent MCPClient] Config file missing: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"[Technical Agent MCPClient] Invalid JSON in config: {e}")
            raise ValueError(f"[Technical Agent MCPClient] Invalid JSON in config: {e}")

    async def fetch_mcp_tools(self) -> None:
        try:
            if not self.mcp_client:
                logger.warning(
                    "[Technical Agent MCPClient] MCP client not initialized, cannot fetch tools"
                )
                raise ValueError(
                    "[Technical Agent MCPClient] MCP client not initialized, cannot fetch tools"
                )

            logger.info(
                f"[Technical Agent MCPClient] Fetching tools - MCP client: {self.mcp_client}"
            )
            tools = await self.mcp_client.get_tools(
                server_name=None
            )  # fetch tools from all servers
            logger.info(f"[Technical Agent MCPClient] Tools fetched: {len(tools)}")
            self.cached_tools = tools  # cache the tools
        except Exception as e:
            logger.error(f"[Technical Agent MCPClient] Error fetching tools: {e}")
            self.cached_tools = None

    async def get_mcp_tools(self) -> list:
        """Return cached tools if available, otherwise fetch from client."""
        if self.cached_tools is not None:
            logger.info("[Technical Agent MCPClient] Returning cached tools")
            return self.cached_tools
        else:
            logger.info(
                "[Technical Agent MCPClient] No cached tools, fetching from MCP client"
            )
            await self.fetch_mcp_tools()
            return self.cached_tools
