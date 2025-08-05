from agents import function_tool

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
