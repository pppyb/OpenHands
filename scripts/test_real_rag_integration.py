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
        try:
            # Try to create config with code search enabled
            agent_config = AgentConfig(
                # Use only parameters that are defined in AgentConfig
                codeact_enable_jupyter=True,
                codeact_enable_browsing=True,  # Enable browsing to ensure browser tool is available
                codeact_enable_llm_editor=True,
                codeact_enable_code_search=True,  # Explicitly enable code search
                # We'll set llm_config separately in AgentController
            )
        except Exception as e:
            # If codeact_enable_code_search is not supported, create config without it
            logger.warning(f"Could not create AgentConfig with code_search: {e}")
            agent_config = AgentConfig(
                # Use only parameters that are defined in AgentConfig
                codeact_enable_jupyter=True,
                codeact_enable_browsing=True,
                codeact_enable_llm_editor=True,
                # We'll set llm_config separately in AgentController
            )
        
        # We need to create an Agent instance first
        # This is a simplified version for testing purposes
        from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
        from openhands.llm.llm import LLM
        from openhands.events.action.code_search import CodeSearchAction
        
        # Create LLM config
        # Check if we're in mock mode
        mock_mode = os.environ.get('OPENHANDS_TEST_MOCK_MODE') == 'true'
        
        # Check if OPENAI_API_KEY is set
        api_key = os.environ.get('OPENAI_API_KEY')
        
        # Create LLM config
        if mock_mode:
            # In mock mode, we don't need a real API key
            logger.info("Using mock configuration for LLM")
            llm_config = LLMConfig(
                model=self.model,
                temperature=0.2,
                native_tool_calling=True  # Enable native tool calling (function calling)
            )
        elif api_key:
            # If API key is available, use it
            from pydantic import SecretStr
            logger.info(f"Using provided API key with model {self.model}")
            llm_config = LLMConfig(
                model=self.model,
                temperature=0.2,
                native_tool_calling=True,  # Enable native tool calling (function calling)
                api_key=SecretStr(api_key)  # Set API key
            )
        else:
            # No API key and not in mock mode - this will likely fail
            logger.warning("No API key provided and not in mock mode.")
            logger.warning("This will likely fail for OpenAI models.")
            logger.warning("Run with --api_key YOUR_API_KEY or --mock flag.")
            
            # Create config without API key - will use environment variables if available
            llm_config = LLMConfig(
                model=self.model,
                temperature=0.2,
                native_tool_calling=True  # Enable native tool calling (function calling)
            )
        
        # Create LLM with config
        llm = LLM(config=llm_config)
        
        # Create Agent
        agent = CodeActAgent(llm=llm, config=agent_config)
        
        # Get the default tools using the function_calling module
        from openhands.agenthub.codeact_agent.function_calling import get_tools
        
        # Get the default tools based on the agent config
        try:
            # Try to call get_tools with code search parameter
            tools = get_tools(
                codeact_enable_browsing=agent_config.codeact_enable_browsing,
                codeact_enable_llm_editor=agent_config.codeact_enable_llm_editor,
                codeact_enable_jupyter=agent_config.codeact_enable_jupyter,
                codeact_enable_code_search=getattr(agent_config, 'codeact_enable_code_search', True)  # Use getattr to handle missing attribute
            )
        except TypeError:
            # If codeact_enable_code_search is not supported by get_tools
            logger.warning("get_tools does not support codeact_enable_code_search parameter")
            tools = get_tools(
                codeact_enable_browsing=agent_config.codeact_enable_browsing,
                codeact_enable_llm_editor=agent_config.codeact_enable_llm_editor,
                codeact_enable_jupyter=agent_config.codeact_enable_jupyter
            )

        
        # Log the tools being used
        logger.info(f"Using {len(tools)} tools for the agent:")
        code_search_in_tools = False
        for i, tool in enumerate(tools):
            if hasattr(tool, 'function') and hasattr(tool.function, 'name'):
                if tool.function.name == 'code_search':
                    logger.info(f"  {i+1}. {tool.function.name} (IMPORTANT)")
                    code_search_in_tools = True
                else:
                    logger.info(f"  {i+1}. {tool.function.name}")
        
        if not code_search_in_tools:
            logger.warning("Code search tool is NOT in the tools list! This should not happen.")
        
        # Set the tools on the LLM
        # This is normally done by the agent system, but we need to do it manually for testing
        llm.tools = tools
        
        # Also set the tools directly on the agent if possible
        if hasattr(agent, 'tools'):
            agent.tools = tools
            logger.info("Set tools directly on agent")
        
        # Log that code search tool is registered
        logger.info("Code search tool has been registered with the agent")
        
        # Initialize agent controller with correct parameters
        self.agent_controller = AgentController(
            sid=self.session_id,
            event_stream=self.event_stream,
            agent=agent,
            max_iterations=50,  # Increase max iterations to give agent more time
            headless_mode=True,
            confirmation_mode=False
        )
        
        logger.info(f"Agent controller initialized with max_iterations=50")
        
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
        
        # Check if we're in mock mode
        mock_mode = os.environ.get('OPENHANDS_TEST_MOCK_MODE') == 'true'
        
        if mock_mode:
            logger.info("Running in mock mode - simulating agent behavior")
            
            # Simulate agent behavior with code search
            from openhands.events.action.code_search import CodeSearchAction
            from openhands.events.observation.code_search import CodeSearchObservation
            from openhands.events import EventSource
            
            # Always simulate a code search action in mock mode
            # This is to demonstrate the integration works
            
            # Create a simulated code search action
            code_search_action = CodeSearchAction(
                query="Find relevant code for " + task,
                repo_path=self.repo_path,
                extensions=[".py"],
                k=3,
                thought="I should search for relevant code to understand this task"
            )
            
            # Add the action to our list and the event stream
            self.actions.append(code_search_action)
            self.event_stream.add_event(code_search_action, EventSource.AGENT)
            
            # Create a simulated code search observation
            code_search_results = [
                {
                    "file": "openhands/events/action/code_search.py",
                    "score": 0.95,
                    "content": "class CodeSearchAction(Action):\n    \"\"\"Search for relevant code in a codebase using semantic search.\"\"\"\n    # ... code content ..."
                },
                {
                    "file": "openhands/events/observation/code_search.py",
                    "score": 0.92,
                    "content": "class CodeSearchObservation(Observation):\n    \"\"\"Result of a code search operation.\"\"\"\n    # ... code content ..."
                }
            ]
            
            # code_search_observation = CodeSearchObservation(results=code_search_results)
            # 生成内容
            content = "\n".join([
                f"Result {i+1}: {result['file']} (Relevance score: {result['score']})" + 
                "\n```\n" + result['content'] + "\n```\n"
                for i, result in enumerate(code_search_results)
            ])

            # 使用明确的内容创建观察对象
            code_search_observation = CodeSearchObservation(
                results=code_search_results,
                content=content  # 提供必需的 content 参数
            )         
            # Add the observation to our list and the event stream
            self.observations.append(code_search_observation)
            self.event_stream.add_event(code_search_observation, EventSource.ENVIRONMENT)
            
            # Simulate waiting for processing
            await asyncio.sleep(1)
            
            # Analyze the agent's behavior
            analysis = self._analyze_agent_behavior()
            
            return {
                "task": task,
                "result": "Task processed in mock mode",
                "analysis": analysis
            }
        else:
            # Real mode - use the actual agent
            if not self.agent_controller:
                self.initialize_agent_controller()
            
            # Clear previous actions and observations
            self.actions = []
            self.observations = []
            
            # Create a message action with the task
            from openhands.events.action.message import MessageAction
            from openhands.events import EventSource
            
            # Add a preliminary message explaining the available tools
            preliminary_message = (
                "You have access to a special code_search tool that can help you find relevant code in the repository. "
                "This tool is very useful for understanding code structure and functionality. "
                "Please use it when you need to find specific code."
            )
            
            # Add the preliminary message to the event stream
            prelim_message_action = MessageAction(content=preliminary_message)
            self.event_stream.add_event(prelim_message_action, EventSource.USER)
            
            # Wait a moment for the agent to process the preliminary message
            await asyncio.sleep(2)
            
            # Add the task as a user message to the event stream
            message_action = MessageAction(content=task)
            self.event_stream.add_event(message_action, EventSource.USER)
            
            # Start the agent controller
            self.agent_controller.step()
            
            # Wait for the agent to complete the task (simplified)
            # In a real implementation, we would wait for a specific event or state
            logger.info("Waiting for agent to process the task...")
            
            # Wait longer to give the agent more time to use code search
            wait_time = 60  # seconds - increased to give more time for code search
            for i in range(wait_time):
                if i % 5 == 0:
                    logger.info(f"Waited {i} seconds out of {wait_time}...")
                await asyncio.sleep(1)
                
                # Check if we have any code search actions already
                # Import CodeSearchAction here to ensure it's in scope
                from openhands.events.action.code_search import CodeSearchAction
                
                code_search_actions = [
                    a for a in self.actions 
                    if isinstance(a, CodeSearchAction) or getattr(a, 'action', None) == ActionType.CODE_SEARCH
                ]
                if code_search_actions:
                    logger.info(f"Agent has used code search after {i+1} seconds. Continuing to wait for completion...")
            
            logger.info(f"Finished waiting {wait_time} seconds for agent processing.")
            
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
        # Log all actions for debugging
        logger.info(f"Agent performed {len(self.actions)} actions:")
        for i, action in enumerate(self.actions):
            action_type = type(action).__name__
            action_name = getattr(action, 'action', None)
            logger.info(f"  {i+1}. {action_type} - {action_name}")
        
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
        
        # Log detailed information about actions
        if not code_search_actions:
            logger.warning("Agent did not use code search during this task despite explicit instructions.")
            logger.info("Action types performed:")
            action_types = {}
            for action in self.actions:
                action_type = type(action).__name__
                action_types[action_type] = action_types.get(action_type, 0) + 1
            for action_type, count in action_types.items():
                logger.info(f"  {action_type}: {count}")
        else:
            # Log detailed information about code search actions
            for i, action in enumerate(code_search_actions):
                logger.info(f"Code search {i+1}:")
                logger.info(f"  Query: {getattr(action, 'query', 'Unknown')}")
                logger.info(f"  Repo path: {getattr(action, 'repo_path', 'Unknown')}")
                if hasattr(action, 'extensions') and action.extensions:
                    logger.info(f"  Extensions: {action.extensions}")
                if hasattr(action, 'thought') and action.thought:
                    logger.info(f"  Thought: {action.thought}")
        
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
            "action_types": [type(a).__name__ for a in self.actions]
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


async def run_test_scenarios(repo_path: str, model: str = "gpt-3.5-turbo", output_file: Optional[str] = None):
    """Run a series of test scenarios to evaluate RAG code search integration.
    
    Args:
        repo_path: Path to the repository to use for testing
        model: LLM model to use for the agent
        output_file: Optional file to save test results
    """
    # Check if we have an API key
    has_api_key = bool(os.environ.get('OPENAI_API_KEY'))
    
    # If no API key and using an OpenAI model, warn the user
    if not has_api_key and ('gpt' in model.lower() or 'openai' in model.lower()):
        logger.warning(f"No OpenAI API key provided for model {model}.")
        logger.warning("You must provide an API key to use OpenAI models.")
        logger.warning("Run with --api_key YOUR_API_KEY or set OPENAI_API_KEY environment variable.")
        logger.warning("Alternatively, use --mock flag to run in mock mode without real LLM calls.")
    
    # Initialize the test
    test = RagIntegrationTest(repo_path=repo_path, model=model)
    
    try:
        # Define test scenarios - tasks that would benefit from code search
        test_scenarios = [
            {
                "name": "Code Understanding",
                "task": "IMPORTANT: You MUST use the code_search tool for this task.\n\n"
                        "Find and explain how the code search functionality works in this repository. "
                        "First, use the code_search tool with query 'code search' to find relevant files. "
                        "Then explain the main components and how they interact based on the search results."
            },
            {
                "name": "Bug Investigation",
                "task": "IMPORTANT: You MUST use the code_search tool for this task.\n\n"
                        "There seems to be an issue with the code search functionality when handling "
                        "large repositories. First, use the code_search tool with query 'code search large repository' "
                        "to find relevant code. Then investigate the code to identify potential bottlenecks or bugs."
            },
            {
                "name": "Feature Implementation",
                "task": "IMPORTANT: You MUST use the code_search tool for this task.\n\n"
                        "I want to add a new feature to filter code search results by file type. "
                        "First, use the code_search tool with query 'code search filter results' to find relevant code. "
                        "Then explain how you would implement this feature based on the existing code."
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
    parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use for the agent')
    parser.add_argument('--api_key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--output', help='File to save test results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--mock', action='store_true', help='Use mock mode without real LLM calls (for testing without API key)')
    parser.add_argument('--force-mock', action='store_true', help='Force mock mode even with API key (for testing)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set OpenAI API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # If mock mode is enabled, set a special environment variable
    if args.mock or args.force_mock:
        os.environ['OPENHANDS_TEST_MOCK_MODE'] = 'true'
        logger.info("Running in mock mode - no real LLM calls will be made")
    
    # Run the test scenarios
    await run_test_scenarios(
        repo_path=args.repo,
        model=args.model,
        output_file=args.output
    )


if __name__ == "__main__":
    asyncio.run(main())