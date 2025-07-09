from crewai import Task
from typing import Dict

def create_recommendation_task(agent, context: Dict):
    return Task(
        description=(
            f"Analyze the user request and provide product recommendations.\n"
            f"User request: {context.get('user_input', 'No input')}\n"
            f"Image context: {context.get('image_description', 'No image')}"
        ),
        agent=agent,
        expected_output="List of 3-5 relevant product recommendations with details.",
        context=context
    )

def create_response_task(agent, context: Dict):
    return Task(
        description=(
            f"Create a helpful response for the user based on:\n"
            f"Products: {context.get('recommendations', [])}\n"
            f"Original request: {context.get('user_input', 'No input')}"
        ),
        agent=agent,
        expected_output="Friendly, informative response summarizing the recommendations.",
        context=context
    )