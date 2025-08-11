import os
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv
from function_tools import get_career_roadmap

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL")
MODEL = "gemini-2.0-flash"


set_tracing_disabled(disabled=True)

client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)
# --- Define Agents ---

# SkillAgent: Specializes in providing skill roadmaps
skill_agent = Agent(
    name="SkillAgent",
    instructions="You are an expert career skills advisor. Your primary role is to provide detailed skill roadmaps for a given career field. Use the `get_career_roadmap` tool when asked about skills or learning paths for a specific career. Once you have provided the roadmap, you can suggest to the user that they might want to learn about job roles or other career paths.",
    tools=[get_career_roadmap], # SkillAgent uses the roadmap tool
    model=OpenAIChatCompletionsModel(model=MODEL,openai_client= client) # Or "gpt-4o", "gpt-3.5-turbo"
)

# JobAgent: Specializes in sharing real-world job roles and responsibilities
job_agent = Agent(
    name="JobAgent",
    instructions="You are an expert job market analyst. Your role is to describe real-world job roles and responsibilities within a specified career field. You can also discuss typical daily tasks, common industries for that role, and entry-level requirements. After providing job role information, you can suggest exploring skills or other career paths.",
    model=OpenAIChatCompletionsModel(model=MODEL,openai_client= client)
)

# CareerAgent: The initial agent that recommends paths and orchestrates handoffs
career_agent = Agent(
    name="CareerAgent",
    instructions=(
        "You are a friendly and encouraging Career Advisor. "
        "If a student elaborates her/his skills, you should handoff to the SkillAgent. "
        "If a student asks about specific job roles or responsibilities, you should handoff to the JobAgent. "
        ),
    # CareerAgent can hand off to SkillAgent and JobAgent
    handoffs=[skill_agent, job_agent],
    tools=[get_career_roadmap],
    model=OpenAIChatCompletionsModel(model=MODEL,openai_client= client)
)

