import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Runner, set_tracing_disabled
from career_agents import career_agent

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL")
MODEL = "gemini-2.0-flash"


set_tracing_disabled(disabled=True)

client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)

    

async def career_exploration_session():
    print("Welcome to the Career Explorer! I'm here to help you find your path.")
    print("What courses have you taken? Tell me about your skills.")

    # Start with the CareerAgent
    current_agent = career_agent
    messages = [] # Keep track of conversation history for context

    while True:
        user_input = input("\nYou: ")
        try:
            if user_input.lower() in ["Thankyou for your guidance","No more questions","Goodbye"]:
                print("AI Agent: Goodbye! Happy exploring!")
                break
            else:
                result = await Runner.run(
                    current_agent,
                    input=user_input,
                    )
                print(f"AI Agent: {result.final_output}")
                messages.append({"role": "assistant", "content": result.final_output})
        except Exception as e:
            print(f"An error occurred: {e}")
            print("AI Agent: I apologize, an error occurred. Please try again or rephrase your query.")

if __name__ == "__main__":
    asyncio.run(career_exploration_session())