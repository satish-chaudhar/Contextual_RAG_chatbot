# Import Crew.AI components for agent-based prompt optimization
from crewai import Agent, Task, Crew

# Import Ollama LLM from langchain_community
from langchain_community.llms import Ollama

# Import os for environment variable access
import os

# Define environment variables for Ollama
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')  # Default Ollama URL
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2:3b')  # Default LLM model

# Initialize Ollama LLM with specified model and URL
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL)

# Define function to optimize either user query or assistant response
def optimize_prompt(user_input: str, assistant_response: str) -> str:
    # Optimize query if user_input is provided
    if user_input:
        # Create agent for query optimization
        agent = Agent(
            role='Query Optimizer',  # Agent's role
            goal='Rephrase user queries to be more clear, concise, and optimized for retrieval in a RAG system.',  # Agent's objective
            backstory='You are an expert in natural language processing and information retrieval.',  # Agent's context
            llm=llm,  # Use Ollama LLM
            verbose=True,  # Enable detailed logging
            allow_delegation=False  # Prevent task delegation
        )
        
        # Define task for query optimization
        task = Task(
            description=f'Rephrase the following user query to improve clarity and relevance for document retrieval: "{user_input}"',  # Task description
            expected_output='A rephrased query string.',  # Expected result
            agent=agent  # Assign task to agent
        )
        
        # Create Crew with single agent and task
        crew = Crew(agents=[agent], tasks=[task], verbose=2)  # Verbose level 2 for detailed logs
        result = crew.kickoff()  # Execute the crew
        return result  # Return optimized query
    
    # Optimize response if assistant_response is provided
    elif assistant_response:
        # Create agent for response optimization
        agent = Agent(
            role='Response Optimizer',  # Agent's role
            goal='Make assistant responses more concise, accurate, and engaging while retaining all key information.',  # Agent's objective
            backstory='You are an expert in communication and summarization.',  # Agent's context
            llm=llm,  # Use Ollama LLM
            verbose=True,  # Enable detailed logging
            allow_delegation=False  # Prevent task delegation
        )
        
        # Define task for response optimization
        task = Task(
            description=f'Optimize the following assistant response to be more concise and clear: "{assistant_response}"',  # Task description
            expected_output='An optimized response string.',  # Expected result
            agent=agent  # Assign task to agent
        )
        
        # Create Crew with single agent and task
        crew = Crew(agents=[agent], tasks=[task], verbose=2)  # Verbose level 2 for detailed logs
        result = crew.kickoff()  # Execute the crew
        return result  # Return optimized response
    
    # Fallback: return input or response if both are empty
    else:
        return user_input or assistant_response