from crewai.flow.flow import Flow,start,listen
from dotenv import load_dotenv,find_dotenv
from litellm import completion
from project.Crews.teaching_crew.teaching_crew import TeachingCrew
from typing import Optional

_: bool = load_dotenv(find_dotenv())

class PanaFlow(Flow):
    """
    A flow that generates trending topics and creates educational content based on those topics.
    Uses Gemini AI for topic generation and TeachingCrew for content creation.
    """
    
    @start()
    def generate_topic(self) -> None:
        """
        Generates a trending topic for 2025 using Gemini AI.
        Stores the topic in the flow's state.
        """
        try:
            response = completion(
                model="gemini/gemini-1.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": """ share the most trending Topic of 2025  """
                    }
                ]
            )
            self.state['topic'] = response['choices'][0]['message']['content']
            print(f" STEP 1 :{self.state['topic']}")
        except Exception as e:
            print(f"Error generating topic: {str(e)}")
            raise

    @listen()
    def generate_content(self) -> None:
        """
        Generates educational content based on the topic using TeachingCrew.
        """
        try:
            print("STEP 2 : Generating Content")
            print("IN GENERATE CONTENT\n")

            if 'topic' not in self.state:
                raise ValueError("No topic found in state. Please run generate_topic first.")

            result = TeachingCrew().crew().kickoff(
                input={
                    "topic": self.state['topic']
                } 
            )

            print(result.raw)
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            raise

def kickoff() -> Optional[str]:
    """
    Initializes and runs the PanaFlow.
    
    Returns:
        Optional[str]: The result of the flow execution, or None if an error occurs
    """
    try:
        flow = PanaFlow()
        return flow.kickoff()
    except Exception as e:
        print(f"Error in flow execution: {str(e)}")
        return None