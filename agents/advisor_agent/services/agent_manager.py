class AgentManager:
    "A registry of all agents/stategraphs in the advisor agent system."

    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_name: str, agent_instance):
        "Register an agent instance with a unique name."
        if agent_name in self.agents:
            raise ValueError(f"Agent with name {agent_name} is already registered.")
        self.agents[agent_name] = agent_instance

    def get_agent(self, agent_name: str):
        "Retrieve an agent instance by name."
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"No agent found with name {agent_name}.")
        return agent


manager = AgentManager()
