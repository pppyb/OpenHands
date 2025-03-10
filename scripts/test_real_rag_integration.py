#!/usr/bin/env python3
"""
Real-world test for RAG code search integration in OpenHands.

This script tests the RAG code search functionality in a real OpenHands agent.
It initializes a CodeActAgent and gives it tasks that would benefit from
code search, then analyzes how the agent uses the code search functionality.
"""

import os
import sys
import json
import argparse
import logging
import asyncio
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import OpenHands components
from openhands.controller.agent_controller import AgentController
from openhands.core.config import AgentConfig, LLMConfig
from openhands.events import EventStream
from openhands.events.action import Action
from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation import Observation
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.core.schema.action import ActionType
from openhands.core.schema.observation import ObservationType
from openhands.storage.local import LocalFileStore

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RagIntegrationTest:
    """Test RAG code search integration in a real OpenHands agent."""
    
    def __init__(self, repo_path: str, model: str = "gpt-4"):
        """Initialize the test.
        
        Args:
            repo_path: Path to the repository to use for testing
            model: LLM model to use for the agent
        """
        self.repo_path = os.path.abspath(repo_path)
        self.model = model
        self.agent_controller = None
        
        # Create a temporary directory for file storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_store_path = self.temp_dir.name
        
        # Create a LocalFileStore
        self.file_store = LocalFileStore(root=self.file_store_path)
        
        # Initialize EventStream with required parameters
        self.session_id = str(uuid.uuid4())
        self.event_stream = EventStream(sid=self.session_id, file_store=self.file_store)
        
        self.actions = []
        self.observations = []
        
    def initialize_agent_controller(self):
        """Initialize the agent controller with a CodeActAgent.
        
        Returns:
            Initialized agent controller
        """
        logger.info(f"Initializing agent controller with repository: {self.repo_path}")
        
        # Create agent config with valid parameters
        agent_config = AgentConfig(
            # Use only parameters that are defined in AgentConfig
            codeact_enable_jupyter=True,
            codeact_enable_browsing=True,  # Enable browsing to ensure browser tool is available
            codeact_enable_llm_editor=True,
            # We'll set llm_config separately in AgentController
        )
        
        # We need to create an Agent instance first
        # This is a simplified version for testing purposes
        from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
        from openhands.llm.llm import LLM
        from openhands.events.action.code_search import CodeSearchAction
        
        # Create LLM config
        llm_config = LLMConfig(
            model=self.model,
            temperature=0.2,
            function_calling=True  # Enable function calling
        )
        
        # Create LLM with config
        llm = LLM(config=llm_config)
        
        # Create Agent
        agent = CodeActAgent(llm=llm, config=agent_config)
        
        # Get the default tools using the function_calling module
        from openhands.agenthub.codeact_agent.function_calling import get_tools
        
        # Get the default tools based on the agent config
        tools = get_tools(
            codeact_enable_browsing=agent_config.codeact_enable_browsing,
            codeact_enable_llm_editor=agent_config.codeact_enable_llm_editor,
            codeact_enable_jupyter=agent_config.codeact_enable_jupyter
        )
        
        # Define the code search tool
        from litellm import ChatCompletionToolParam
        
        code_search_tool = ChatCompletionToolParam(
            type="function",
            function={
                "name": "code_search",
                "description": "Search for relevant code in a codebase using semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search for code."
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository to search."
                        },
                        "extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file extensions to include (e.g. [\".py\", \".js\"])."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return."
                        },
                        "thought": {
                            "type": "string",
                            "description": "Reasoning behind the search."
                        }
                    },
                    "required": ["query"]
                }
            }
        )
        
        # Add the code search tool to the tools list
        tools.append(code_search_tool)
        
        # Set the tools on the LLM
        # This is normally done by the agent system, but we need to do it manually for testing
        llm.tools = tools
        
        # Initialize agent controller with correct parameters
        self.agent_controller = AgentController(
            sid=self.session_id,
            event_stream=self.event_stream,
            agent=agent,
            max_iterations=20,
            headless_mode=True,
            confirmation_mode=False
        )
        
        # Subscribe to events with a unique callback_id
        self.event_stream.subscribe("test", self.event_callback, "test_callback")
        
        return self.agent_controller
    
    def event_callback(self, event):
        """Callback for events from the agent.
        
        Args:
            event: Event from the agent
        """
        # Store actions and observations
        if hasattr(event, 'action') and event.action:
            self.actions.append(event.action)
        if hasattr(event, 'observation') and event.observation:
            self.observations.append(event.observation)
    
    async def run_task(self, task: str) -> Dict[str, Any]:
        """Run a task with the agent and collect actions and observations.
        
        Args:
            task: Task description for the agent to execute
            
        Returns:
            Dictionary with task results and analysis
        """
        logger.info(f"Running task: {task}")
        
        if not self.agent_controller:
            self.initialize_agent_controller()
        
        # Clear previous actions and observations
        self.actions = []
        self.observations = []
        
        # Create a message action with the task
        from openhands.events.action.message import MessageAction
        from openhands.events import EventSource
        
        # Add the task as a user message to the event stream
        message_action = MessageAction(content=task)
        self.event_stream.add_event(message_action, EventSource.USER)
        
        # Start the agent controller
        self.agent_controller.step()
        
        # Wait for the agent to complete the task (simplified)
        # In a real implementation, we would wait for a specific event or state
        await asyncio.sleep(10)  # Wait a bit for the agent to process
        
        # Analyze the agent's behavior
        analysis = self._analyze_agent_behavior()
        
        return {
            "task": task,
            "result": "Task processed",
            "analysis": analysis
        }
    
    def _analyze_agent_behavior(self) -> Dict[str, Any]:
        """Analyze how the agent used code search during the task.
        
        Returns:
            Dictionary with analysis results
        """
        # Find all code search actions
        code_search_actions = [
            a for a in self.actions 
            if isinstance(a, CodeSearchAction) or getattr(a, 'action', None) == ActionType.CODE_SEARCH
        ]
        
        # Find all code search observations
        code_search_observations = [
            o for o in self.observations 
            if isinstance(o, CodeSearchObservation) or getattr(o, 'observation', None) == ObservationType.CODE_SEARCH
        ]
        
        # Match actions with their observations
        action_observation_pairs = []
        for action in code_search_actions:
            # Find the observation that was caused by this action
            matching_obs = next(
                (o for o in code_search_observations if getattr(o, "cause", None) == getattr(action, "id", None)), 
                None
            )
            
            action_observation_pairs.append({
                "action": {
                    "id": getattr(action, "id", None),
                    "query": getattr(action, "query", None),
                    "thought": getattr(action, "thought", ""),
                },
                "observation": {
                    "id": getattr(matching_obs, "id", None) if matching_obs else None,
                    "results_count": len(getattr(matching_obs, "results", [])) if matching_obs else 0,
                } if matching_obs else None
            })
        
        return {
            "code_search_count": len(code_search_actions),
            "action_observation_pairs": action_observation_pairs,
            "total_actions": len(self.actions),
            "code_search_percentage": len(code_search_actions) / len(self.actions) if self.actions else 0,
        }
    
    def get_detailed_report(self) -> str:
        """Generate a detailed report of the agent's code search usage.
        
        Returns:
            Formatted string with detailed report
        """
        if not self.actions:
            return "No actions recorded. Run a task first."
        
        # Find all code search actions
        code_search_actions = [
            a for a in self.actions 
            if isinstance(a, CodeSearchAction) or getattr(a, 'action', None) == ActionType.CODE_SEARCH
        ]
        
        if not code_search_actions:
            return "Agent did not use code search during this task."
        
        report = ["## Code Search Usage Report", ""]
        report.append(f"Total actions: {len(self.actions)}")
        report.append(f"Code search actions: {len(code_search_actions)} ({len(code_search_actions)/len(self.actions):.1%})")
        report.append("")
        
        for i, action in enumerate(code_search_actions, 1):
            report.append(f"### Code Search {i}")
            report.append(f"Query: \"{getattr(action, 'query', 'Unknown')}\"")
            
            if hasattr(action, "thought") and action.thought:
                report.append(f"Thought: {action.thought}")
            
            # Find the observation for this action
            matching_obs = next(
                (o for o in self.observations if getattr(o, "cause", None) == getattr(action, "id", None)), 
                None
            )
            
            if matching_obs and hasattr(matching_obs, "results"):
                report.append(f"Results: {len(matching_obs.results)}")
                
                for j, result in enumerate(matching_obs.results[:3], 1):  # Show top 3 results
                    report.append(f"  Result {j}: {result['file']} (Score: {result['score']:.3f})")
                    report.append("  ```")
                    # Truncate content if too long
                    content = result['content']
                    if len(content) > 300:
                        content = content[:300] + "..."
                    report.append(f"  {content}")
                    report.append("  ```")
                
                if len(matching_obs.results) > 3:
                    report.append(f"  ... and {len(matching_obs.results) - 3} more results")
            
            report.append("")
        
        return "\n".join(report)


async def run_test_scenarios(repo_path: str, model: str = "gpt-4", output_file: Optional[str] = None):
    """Run a series of test scenarios to evaluate RAG code search integration.
    
    Args:
        repo_path: Path to the repository to use for testing
        model: LLM model to use for the agent
        output_file: Optional file to save test results
    """
    # Initialize the test
    test = RagIntegrationTest(repo_path=repo_path, model=model)
    
    try:
        # Define test scenarios - tasks that would benefit from code search
        test_scenarios = [
            {
                "name": "Code Understanding",
                "task": "Explain how the code search functionality works in this repository. "
                        "What are the main components and how do they interact?"
            },
            {
                "name": "Bug Investigation",
                "task": "There seems to be an issue with the code search functionality when handling "
                        "large repositories. Investigate the code to find potential bottlenecks or bugs."
            },
            {
                "name": "Feature Implementation",
                "task": "I want to add a new feature to filter code search results by file type. "
                        "How would you implement this based on the existing code?"
            }
        ]
        
        # Run each test scenario
        results = []
        for scenario in test_scenarios:
            logger.info(f"Running test scenario: {scenario['name']}")
            
            result = await test.run_task(scenario['task'])
            detailed_report = test.get_detailed_report()
            
            scenario_result = {
                "scenario": scenario,
                "result": result,
                "detailed_report": detailed_report
            }
            
            results.append(scenario_result)
            
            # Print the detailed report
            print(f"\n{'='*80}")
            print(f"Test Scenario: {scenario['name']}")
            print(f"{'='*80}")
            print(detailed_report)
        
        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {output_file}")
        
        return results
    finally:
        # Clean up temporary directory
        if hasattr(test, 'temp_dir'):
            test.temp_dir.cleanup()
            logger.info("Cleaned up temporary directory")


async def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description='Test RAG code search integration in a real OpenHands agent')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to use for testing')
    parser.add_argument('--model', default='gpt-4', help='LLM model to use for the agent')
    parser.add_argument('--output', help='File to save test results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the test scenarios
    await run_test_scenarios(
        repo_path=args.repo,
        model=args.model,
        output_file=args.output
    )


if __name__ == "__main__":
    asyncio.run(main())