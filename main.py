import os
from typing import Dict, List, TypedDict, Literal
import json

from langgraph.graph import StateGraph, END
import anthropic
from openai import OpenAI

# Import agents
from agents.operational_agent import OperationalAgent
from agents.evaluative_agent import EvaluativeAgent
from agents.advisory_agent import AdvisoryAgent
from memory.feedback_logger import FeedbackLogger

# Import utilities
from utils.dicom_handler import DicomHandler
from utils.dose_analysis import DoseAnalyzer
from utils.visualization import Visualizer


# Define the application state
class PlanningState(TypedDict):
    ct_data: Dict
    structures: Dict
    objectives: Dict
    plan_parameters: Dict
    plan_results: Dict
    evaluation_results: Dict
    recommendations: List[Dict]
    iteration: int
    status: Literal["planning", "evaluating", "advising", "complete", "human_review"]
    memory: List[Dict]


# Initialize clients
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize components
dicom_handler = DicomHandler()
dose_analyzer = DoseAnalyzer()
visualizer = Visualizer()
feedback_logger = FeedbackLogger("memory/planning_history.json")

# Initialize agents
operational_agent = OperationalAgent(openai_client, dicom_handler)
evaluative_agent = EvaluativeAgent(claude_client, dose_analyzer, visualizer)
advisory_agent = AdvisoryAgent(claude_client, feedback_logger)


def initialize_planning(state: PlanningState) -> PlanningState:
    """Initialize the treatment planning process"""
    # Log the start of a new planning session
    feedback_logger.log_event(
        "system", "Started new planning session", {"iteration": state["iteration"]}
    )

    return {**state, "status": "planning"}


def run_operational_agent(state: PlanningState) -> PlanningState:
    """Run the operational agent to set parameters and execute planning"""
    results = operational_agent.execute(
        state["ct_data"],
        state["structures"],
        state["objectives"],
        state["plan_parameters"] if state["iteration"] > 1 else None,
        state["recommendations"] if state["iteration"] > 1 else [],
    )

    # Log the parameters used
    feedback_logger.log_event(
        "operational",
        "Set planning parameters",
        {"parameters": results["plan_parameters"], "iteration": state["iteration"]},
    )

    return {
        **state,
        "plan_parameters": results["plan_parameters"],
        "plan_results": results["plan_results"],
        "status": "evaluating",
    }


def run_evaluative_agent(state: PlanningState) -> PlanningState:
    """Run the evaluative agent to assess the treatment plan"""
    evaluation = evaluative_agent.execute(
        state["ct_data"], state["structures"], state["objectives"], state["plan_results"]
    )

    # Log the evaluation results
    feedback_logger.log_event(
        "evaluative",
        "Evaluated plan quality",
        {"evaluation": evaluation, "iteration": state["iteration"]},
    )

    # Update state based on evaluation outcome
    new_status = "complete" if evaluation["is_acceptable"] else "advising"

    # Check if human review is needed
    if evaluation["needs_human_review"]:
        new_status = "human_review"

    return {**state, "evaluation_results": evaluation, "status": new_status}


def run_advisory_agent(state: PlanningState) -> PlanningState:
    """Run the advisory agent to suggest improvements"""
    recommendations = advisory_agent.execute(
        state["objectives"],
        state["plan_parameters"],
        state["plan_results"],
        state["evaluation_results"],
        state["memory"],
    )

    # Log the recommendations
    feedback_logger.log_event(
        "advisory",
        "Generated recommendations",
        {"recommendations": recommendations, "iteration": state["iteration"]},
    )

    # Prepare for next iteration
    next_iteration = state["iteration"] + 1

    return {
        **state,
        "recommendations": recommendations,
        "iteration": next_iteration,
        "memory": state["memory"]
        + [
            {
                "iteration": state["iteration"],
                "plan_parameters": state["plan_parameters"],
                "evaluation_results": state["evaluation_results"],
                "recommendations": recommendations,
            }
        ],
        "status": "planning",
    }


def human_review(state: PlanningState) -> PlanningState:
    """Process for human expert review"""
    print("\n==== PLAN REQUIRES HUMAN REVIEW ====")
    print(f"Iteration: {state['iteration']}")
    print(f"Evaluation results: {json.dumps(state['evaluation_results'], indent=2)}")

    # Here you would implement your human-in-the-loop interface
    # For this POC, we'll simulate a simple CLI interface

    decision = input("Accept plan? (yes/no/modify): ").strip().lower()

    if decision == "yes":
        return {**state, "status": "complete"}
    elif decision == "no":
        # Get recommendations from the advisory agent
        return run_advisory_agent(state)
    else:  # modify
        # Allow manual parameter adjustments
        print("Current parameters:", json.dumps(state["plan_parameters"], indent=2))
        print("Enter modifications (empty to skip):")

        # This is a simplified example - you would need a more robust interface
        modified_params = state["plan_parameters"].copy()

        # Example manual modification
        for key in modified_params:
            new_value = input(f"{key} (current: {modified_params[key]}): ")
            if new_value:
                # Convert to appropriate type
                if isinstance(modified_params[key], int):
                    modified_params[key] = int(new_value)
                elif isinstance(modified_params[key], float):
                    modified_params[key] = float(new_value)
                else:
                    modified_params[key] = new_value

        recommendations = [
            {
                "source": "human",
                "suggestion": "Manual parameter adjustment",
                "parameters": modified_params,
            }
        ]

        feedback_logger.log_event(
            "human",
            "Manual parameter adjustment",
            {"modifications": modified_params, "iteration": state["iteration"]},
        )

        return {
            **state,
            "recommendations": recommendations,
            "iteration": state["iteration"] + 1,
            "memory": state["memory"]
            + [
                {
                    "iteration": state["iteration"],
                    "plan_parameters": state["plan_parameters"],
                    "evaluation_results": state["evaluation_results"],
                    "recommendations": recommendations,
                    "source": "human",
                }
            ],
            "status": "planning",
        }


def router(
    state: PlanningState,
) -> Literal["operational", "evaluative", "advisory", "human_review", "end"]:
    """Route to the next step based on the current status"""
    if state["status"] == "planning":
        return "operational"
    elif state["status"] == "evaluating":
        return "evaluative"
    elif state["status"] == "advising":
        return "advisory"
    elif state["status"] == "human_review":
        return "human_review"
    else:  # complete
        return "end"


def create_planning_workflow():
    """Create the LangGraph workflow for the treatment planning system"""
    # Initialize the graph
    workflow = StateGraph(PlanningState)

    # Add nodes
    workflow.add_node("initialization", initialize_planning)
    workflow.add_node("operational", run_operational_agent)
    workflow.add_node("evaluative", run_evaluative_agent)
    workflow.add_node("advisory", run_advisory_agent)
    workflow.add_node("human_review", human_review)

    # Add edges
    workflow.add_edge("initialization", "operational")
    workflow.add_conditional_edges("operational", router, {"evaluative": "evaluative", "end": END})
    workflow.add_conditional_edges(
        "evaluative", router, {"advisory": "advisory", "human_review": "human_review", "end": END}
    )
    workflow.add_conditional_edges("advisory", router, {"operational": "operational", "end": END})
    workflow.add_conditional_edges(
        "human_review", router, {"operational": "operational", "advisory": "advisory", "end": END}
    )

    # Compile the graph
    return workflow.compile()


def run_planning_system(ct_folder_path, structure_file_path, objectives_file_path):
    """Run the complete treatment planning workflow"""
    # Load input data
    ct_data = dicom_handler.load_ct_series(ct_folder_path)
    structures = dicom_handler.load_structures(structure_file_path)
    with open(objectives_file_path, "r") as f:
        objectives = json.load(f)

    # Initialize state
    initial_state = PlanningState(
        ct_data=ct_data,
        structures=structures,
        objectives=objectives,
        plan_parameters={},
        plan_results={},
        evaluation_results={},
        recommendations=[],
        iteration=1,
        status="planning",
        memory=[],
    )

    # Create and run the workflow
    workflow = create_planning_workflow()
    final_state = workflow.invoke({"state": initial_state})

    print("\n==== PLANNING COMPLETE ====")
    print(f"Completed after {final_state['iteration']} iterations")
    print(f"Final plan parameters: {json.dumps(final_state['plan_parameters'], indent=2)}")
    print(f"Final evaluation: {json.dumps(final_state['evaluation_results'], indent=2)}")

    # Save results
    results_folder = f"results/plan_{int(time.time())}"
    os.makedirs(results_folder, exist_ok=True)

    # Save plan parameters and results
    with open(f"{results_folder}/plan_parameters.json", "w") as f:
        json.dump(final_state["plan_parameters"], f, indent=2)

    with open(f"{results_folder}/evaluation_results.json", "w") as f:
        json.dump(final_state["evaluation_results"], f, indent=2)

    # Save dose visualizations
    visualizer.save_dose_visualizations(
        final_state["ct_data"],
        final_state["structures"],
        final_state["plan_results"]["dose_distribution"],
        results_folder,
    )

    return final_state


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="IMRT Treatment Planning System")
    parser.add_argument("--ct_folder", type=str, required=True, help="Path to CT DICOM folder")
    parser.add_argument("--structure_file", type=str, required=True, help="Path to RTSTRUCT file")
    parser.add_argument(
        "--objectives_file", type=str, required=True, help="Path to treatment objectives JSON file"
    )

    args = parser.parse_args()

    final_state = run_planning_system(args.ct_folder, args.structure_file, args.objectives_file)
