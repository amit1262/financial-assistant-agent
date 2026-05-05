# manages mcp clients for all the agents in the market intelligence unit.


class MCPClientManager:
    """Class to manage MCP clients for all agents in the market intelligence unit."""

    def __init__(self):
        self.clients = {}

    def get_client(self, agent_name):
        "get mcp client for the given agent name."
        client = self.clients.get(agent_name, None)
        if client is None:
            raise ValueError(f"No MCP client found for agent: {agent_name}")
        return client

    def register_client(self, agent_name, client):
        "register mcp client for the given agent name."
        if agent_name in self.clients:
            raise ValueError(f"MCP client already registered for agent: {agent_name}")
        self.clients[agent_name] = client


mcp_manager = (
    MCPClientManager()
)  # create a global instance of the MCPClientManager class
