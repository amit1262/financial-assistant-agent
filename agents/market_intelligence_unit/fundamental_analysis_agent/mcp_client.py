# this module utilizes the user-defined MCP servers configs to create MCP clients
import json
import os
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(
        self, servers: dict, client: MultiServerMCPClient, tool_name_prefix: bool = True
    ):
        self.servers = servers
        self.client = client
        self.tool_name_prefix = tool_name_prefix
        self._cached_tools = None  # cache to store fetched tools

    @classmethod
    async def create(cls, tool_name_prefix: bool = True):
        mcp_config_path = Path(__file__).parent / "mcp-config.json"
        servers = await cls._get_servers(mcp_config_path)
        client = await cls._create_client(
            servers=servers, tool_name_prefix=tool_name_prefix
        )
        instance = cls(
            servers=servers, client=client, tool_name_prefix=tool_name_prefix
        )
        await instance.fetch_tools()  # fetch and cache tools on initialization
        return instance

    async def _create_client(servers: dict, tool_name_prefix):
        """Create the MCP client instance."""
        if not servers:
            logger.warning(
                "[Fundamental Agent] No servers available to create MCP client"
            )
            raise ValueError(
                "[Fundamental Agent] No servers available to create MCP client"
            )
        try:
            client = MultiServerMCPClient(
                connections=servers, tool_name_prefix=tool_name_prefix
            )
            logger.info("[Fundamental Agent] MCP Client successfully created")
            return client
        except Exception as e:
            logger.error(f"[Fundamental Agent] Error creating MCP Client: {e}")
            raise ValueError(f"[Fundamental Agent] Error creating MCP Client: {e}")

    async def _get_servers(mcp_config_path) -> dict:
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
                    "[Fundamental Agent] No servers configured in mcp-config.json"
                )
                raise ValueError(
                    "[Fundamental Agent] No servers configured in mcp-config.json"
                )

            # Substitute API keys
            for server_name, server_config in servers_dict.items():
                url = server_config.get("url", "")

                if "API_KEY" in url:
                    env_key_name = f"{server_name.upper()}_API_KEY_FA"
                    api_key = os.getenv(env_key_name)

                    if not api_key:
                        logger.error(
                            f"[Fundamental Agent] Missing API key for '{server_name}': set {env_key_name} env var"
                        )
                        raise ValueError(
                            f"[Fundamental Agent] Environment variable not set: {env_key_name}"
                        )
                    else:
                        server_config["url"] = url.format(API_KEY=api_key)
                        final_dict[server_name] = server_config
                    # logger.info(f"API key substituted for server: {server_name}")

            logger.info(f"MCP servers: {len(final_dict)}")
            return final_dict

        except FileNotFoundError as e:
            logger.error(f"[Fundamental Agent] Config file missing: {e}")
            raise ValueError(f"[Fundamental Agent] Config file missing: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"[Fundamental Agent] Invalid JSON in config: {e}")
            raise ValueError(f"[Fundamental Agent] Invalid JSON in config: {e}")

    async def fetch_tools(self, server_name: str = None) -> None:
        try:
            if not self.client:
                logger.warning(
                    "[Fundamental Agent] MCP client not initialized, cannot fetch tools"
                )
                raise ValueError(
                    "[Fundamental Agent] MCP client not initialized, cannot fetch tools"
                )

            logger.info(
                f"[Fundamental Agent] Fetching tools - MCP client: {self.client}, Server: {server_name or 'all servers'}"
            )
            tools = await self.client.get_tools(server_name=server_name)
            logger.info(f"[Fundamental Agent] Tools fetched: {tools}")
            self._cached_tools = tools  # cache the tools
        except Exception as e:
            logger.error(f"[Fundamental Agent] Error fetching tools: {e}")
            self._cached_tools = None

    async def get_mcp_tools(self) -> list:
        """Return cached tools if available, otherwise fetch from client."""
        if self._cached_tools is not None:
            logger.info("[Fundamental Agent] Returning cached tools")
            return self._cached_tools
        else:
            logger.info("[Fundamental Agent] No cached tools, fetching from MCP client")
            await self.fetch_tools()
            return self._cached_tools
