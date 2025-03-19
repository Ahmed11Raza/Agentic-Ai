from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew
import yaml

@CrewBase
class TeachingCrew:
    agent_config = yaml.safe_load(open("config/config.agent.yaml"))
    task_config = yaml.safe_load(open("config/config.task.yaml"))

    @agent
    def sir_zia_agent(self) -> Agent:
        return Agent(
            config=self.agent_config["sir_zia"]
        )

    @task
    def describe_Topic(self) -> Task:
        return Task(
            config=self.task_config["describe_Topic"],
            input_args=["topic"],
            agent=self.sir_zia_agent  # Ensure the task is assigned to the agent
        )    
    
    @crew
    def crew(self) -> Crew:
        """
        Creates and returns a teaching crew that will generate educational content.
        
        Returns:
            Crew: The configured teaching crew
        """
        return Crew(
            agents=[self.sir_zia_agent()],
            tasks=[self.describe_Topic()],
            verbose=True
        )
        
        
        
        
    


