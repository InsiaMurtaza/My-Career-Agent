import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

set_tracing_disabled(disabled=True)

client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)

@function_tool
def get_career_roadmap(career_field: str) -> str:
    """
    Generates a skill roadmap and learning path for a given career field.
    Args:
        career_field (str): The specific career field for which to generate a roadmap (e.g., "Software Engineer", "Data Scientist", "Digital Marketer").
    Returns:
        str: A detailed string describing the essential skills, recommended learning resources, and a suggested progression for the chosen career field.
    """
    if "software engineer" in career_field.lower():
        return (
            "**Software Engineer Roadmap:**\n"
            "1. **Core Programming:** Python/Java/C++ fundamentals, data structures, algorithms.\n"
            "2. **Web Development (Frontend/Backend):** HTML, CSS, JavaScript, React/Angular/Vue, Node.js/Django/Spring Boot.\n"
            "3. **Databases:** SQL (PostgreSQL/MySQL), NoSQL (MongoDB/Redis).\n"
            "4. **Version Control:** Git, GitHub/GitLab.\n"
            "5. **DevOps Basics:** Docker, CI/CD.\n"
            "6. **Advanced Topics:** Cloud platforms (AWS/Azure/GCP), system design, microservices.\n"
            "**Learning Resources:** LeetCode, FreeCodeCamp, Coursera, Udemy, official docs."
        )
    elif "data scientist" in career_field.lower():
        return (
            "**Data Scientist Roadmap:**\n"
            "1. **Mathematics & Statistics:** Linear algebra, calculus, probability, inferential statistics.\n"
            "2. **Programming:** Python (Pandas, NumPy, Scikit-learn), R.\n"
            "3. **Machine Learning:** Supervised/Unsupervised learning, deep learning (TensorFlow/PyTorch).\n"
            "4. **Data Manipulation & Visualization:** SQL, Tableau/Matplotlib/Seaborn.\n"
            "5. **Big Data Technologies:** Spark, Hadoop (optional).\n"
            "6. **Communication & Storytelling:** Presenting insights clearly.\n"
            "**Learning Resources:** Kaggle, DataCamp, Udacity, specialized university courses."
        )
    elif "digital marketer" in career_field.lower():
        return (
            "**Digital Marketer Roadmap:**\n"
            "1. **Marketing Fundamentals:** Principles of marketing, market research.\n"
            "2. **SEO (Search Engine Optimization):** Keyword research, on-page/off-page SEO.\n"
            "3. **SEM (Search Engine Marketing):** Google Ads, PPC strategies.\n"
            "4. **Social Media Marketing:** Platform-specific strategies, content creation.\n"
            "5. **Content Marketing:** Blogging, video content, copywriting.\n"
            "6. **Email Marketing:** CRM tools, campaign management.\n"
            "7. **Analytics:** Google Analytics, data interpretation.\n"
            "**Learning Resources:** Google Digital Garage, HubSpot Academy, industry blogs."
        )
    else:
        return f"Sorry, I don't have a specific roadmap for '{career_field}' yet. Please choose from 'Software Engineer', 'Data Scientist', or 'Digital Marketer'."
    
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
                    # messages=messages, # Pass the entire conversation history
                )
                print(f"AI Agent: {result.final_output}")
                messages.append({"role": "assistant", "content": result.final_output})
        except Exception as e:
            print(f"An error occurred: {e}")
            print("AI Agent: I apologize, an error occurred. Please try again or rephrase your query.")
            # Depending on error, you might want to reset the agent or provide more specific guidance.

if __name__ == "__main__":
    asyncio.run(career_exploration_session())