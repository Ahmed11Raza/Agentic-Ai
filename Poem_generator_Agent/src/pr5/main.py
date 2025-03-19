from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from pr5.crews.poem_crew.poem_crew import PoemCrew

class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""

class PoemFlow(Flow[PoemState]):
    @start()
    def generate_sentence_count(self):
        print("Enter the number of sentences (1-5): ")
        while True:
            try:
                count = int(input())
                if 1 <= count <= 5:
                    self.state.sentence_count = count
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    @listen(generate_sentence_count)
    def generate_poem(self):
        print("Generating poem")
        try:
            result = (PoemCrew().crew().kickoff(inputs={"sentence_count": self.state.sentence_count}))
            self.state.poem = result.raw
            print("\nGenerated Poem:\n")
            print(self.state.poem)
        except Exception as e:
            print(f"Error generating poem: {e}")

    @listen(generate_poem)
    def save_poem(self):
        print("Do you want to save the poem? (yes/no): ")
        while True:
            response = input().lower()
            if response == "yes":
                filename = input("Enter the filename to save the poem: ")
                try:
                    with open(filename, "w") as f:
                        f.write(self.state.poem)
                        print(f"Poem saved to {filename}")
                    break
                except Exception as e:
                    print(f"Error saving poem: {e}")
            elif response == "no":
                print("Poem not saved.")
                break
            else:
                print("Invalid response. Please enter 'yes' or 'no'.")

def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()

def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()

if __name__ == "__main__":
    kickoff()