#!/usr/bin/env python3
"""
Test RAG code search integration in a real OpenHands agent.

This script tests how the RAG code search functionality is integrated and used
in a real OpenHands agent. It initializes an agent with a specific repository,
gives it tasks that require code understanding, and analyzes how the agent
uses the code search functionality to complete these tasks.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import OpenHands components
from openhands.agent import Agent
from openhands.events.action import Action
from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation import Observation
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.core.schema.action import ActionType
from openhands.core.schema.observation import ObservationType

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RagTestAgent:
    """Test agent that uses RAG code search functionality."""
    
    def __init__(self, repo_path: str, model: str = "gpt-4"):
        """Initialize the test agent.
        
        Args:
            repo_path: Path to the repository to use as the agent's workspace
            model: LLM model to use for the agent
        """
        self.repo_path = os.path.abspath(repo_path)
        self.model = model
        self.agent = self._initialize_agent()
        self.actions: List[Action] = []
        self.observations: List[Observation] = []
        
    def _initialize_agent(self) -> Agent:
        """Initialize the OpenHands agent with the specified repository.
        
        Returns:
            Initialized OpenHands agent
        """
        logger.info(f"Initializing agent with repository: {self.repo_path}")
        
        # Initialize the agent with the repository path
        agent = Agent(
            model=self.model,
            work_dir=self.repo_path,
        )
        
        return agent
    
    def run_task(self, task: str) -> Dict[str, Any]:
        """Run a task with the agent and collect actions and observations.
        
        Args:
            task: Task description for the agent to execute
            
        Returns:
            Dictionary with task results and analysis
        """
        logger.info(f"Running task: {task}")
        
        # Run the agent with the task
        result = self.agent.run(task)
        
        # Collect actions and observations
        self.actions = self.agent.get_actions()
        self.observations = self.agent.get_observations()
        
        # Analyze the agent's behavior
        analysis = self._analyze_agent_behavior()
        
        return {
            "task": task,
            "result": result,
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
            if isinstance(a, CodeSearchAction) or a.action == ActionType.CODE_SEARCH
        ]
        
        # Find all code search observations
        code_search_observations = [
            o for o in self.observations 
            if isinstance(o, CodeSearchObservation) or o.observation == ObservationType.CODE_SEARCH
        ]
        
        # Match actions with their observations
        action_observation_pairs = []
        for action in code_search_actions:
            # Find the observation that was caused by this action
            matching_obs = next(
                (o for o in code_search_observations if getattr(o, "cause", None) == action.id), 
                None
            )
            
            action_observation_pairs.append({
                "action": {
                    "id": action.id,
                    "query": action.query,
                    "thought": getattr(action, "thought", ""),
                },
                "observation": {
                    "id": matching_obs.id if matching_obs else None,
                    "results_count": len(matching_obs.results) if matching_obs else 0,
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
            if isinstance(a, CodeSearchAction) or a.action == ActionType.CODE_SEARCH
        ]
        
        if not code_search_actions:
            return "Agent did not use code search during this task."
        
        report = ["## Code Search Usage Report", ""]
        report.append(f"Total actions: {len(self.actions)}")
        report.append(f"Code search actions: {len(code_search_actions)} ({len(code_search_actions)/len(self.actions):.1%})")
        report.append("")
        
        for i, action in enumerate(code_search_actions, 1):
            report.append(f"### Code Search {i}")
            report.append(f"Query: \"{action.query}\"")
            
            if hasattr(action, "thought") and action.thought:
                report.append(f"Thought: {action.thought}")
            
            # Find the observation for this action
            matching_obs = next(
                (o for o in self.observations if getattr(o, "cause", None) == action.id), 
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


def run_test_scenarios(repo_path: str, model: str = "gpt-4", output_file: Optional[str] = None):
    """Run a series of test scenarios to evaluate RAG code search integration.
    
    Args:
        repo_path: Path to the repository to use for testing
        model: LLM model to use for the agent
        output_file: Optional file to save test results
    """
    # Initialize the test agent
    test_agent = RagTestAgent(repo_path=repo_path, model=model)
    
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
        
        result = test_agent.run_task(scenario['task'])
        detailed_report = test_agent.get_detailed_report()
        
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


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description='Test RAG code search integration in OpenHands agent')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to use for testing')
    parser.add_argument('--model', default='gpt-4', help='LLM model to use for the agent')
    parser.add_argument('--output', help='File to save test results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the test scenarios
    run_test_scenarios(
        repo_path=args.repo,
        model=args.model,
        output_file=args.output
    )


if __name__ == "__main__":
    main()