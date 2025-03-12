"""Base class for action handlers."""

from abc import ABC, abstractmethod
from typing import Optional

from openhands.events.action import Action
from openhands.events.observation import Observation


class ActionHandler(ABC):
    """Base class for action handlers."""

    @abstractmethod
    def can_handle(self, action: Action) -> bool:
        """Check if this handler can handle the given action.
        
        Args:
            action: The action to check
            
        Returns:
            True if this handler can handle the action, False otherwise
        """
        pass

    @abstractmethod
    def handle(self, action: Action) -> Optional[Observation]:
        """Handle an action.
        
        Args:
            action: The action to handle
            
        Returns:
            An observation resulting from handling the action, or None if the action
            could not be handled
        """
        pass