"""
Workflow management for multi-step processes
"""

import streamlit as st
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manage multi-step workflow state"""
    
    def __init__(self):
        """Initialize workflow manager"""
        self.initialize_state()
    
    def initialize_state(self):
        """Initialize workflow state in session"""
        if 'workflow_state' not in st.session_state:
            st.session_state.workflow_state = {
                'current_step': 1,
                'completed_steps': [],
                'data': {}
            }
    
    def advance_step(self):
        """Advance to next workflow step"""
        current = st.session_state.workflow_state['current_step']
        st.session_state.workflow_state['completed_steps'].append(current)
        st.session_state.workflow_state['current_step'] = current + 1
        logger.info(f"Advanced to step {current + 1}")
    
    def can_proceed(self, step: int) -> bool:
        """Check if user can proceed to a step"""
        return step <= st.session_state.workflow_state['current_step']
    
    def save_step_data(self, step: int, data: Any):
        """Save data for a workflow step"""
        st.session_state.workflow_state['data'][f'step_{step}'] = data
    
    def get_step_data(self, step: int) -> Any:
        """Get data for a workflow step"""
        return st.session_state.workflow_state['data'].get(f'step_{step}')
    
    def reset_workflow(self):
        """Reset workflow to initial state"""
        st.session_state.workflow_state = {
            'current_step': 1,
            'completed_steps': [],
            'data': {}
        }
        logger.info("Workflow reset")
    
    def get_progress(self) -> float:
        """Get workflow progress percentage"""
        total_steps = 5  # Total workflow steps
        current = st.session_state.workflow_state['current_step']
        return min(current / total_steps, 1.0)