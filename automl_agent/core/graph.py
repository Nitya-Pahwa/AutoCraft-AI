import os
from typing import Literal

from langgraph.graph import END, StateGraph
from automl_agent.agents.analyst import analyst_agent
from automl_agent.agents.critic import critic_agent
from automl_agent.agents.evaluator import evaluator_agent
from automl_agent.agents.planner import planner_agent
from automl_agent.agents.reporter import reporter_agent
from automl_agent.agents.selector import selector_agent
from automl_agent.agents.trainer import trainer_agent
from automl_agent.core.state import AgentState, DatasetInfo
from automl_agent.utils.memory import memory_retrieve_agent, memory_store_agent

GRAPH_SAVE_PATH = os.path.join("artifacts", "pipeline_graph.png")


def _coerce_state(state: AgentState | dict) -> AgentState:
    if isinstance(state, AgentState):
        return state
    return AgentState(**state)


def _wrap(fn):
    def wrapped(state_dict: AgentState | dict) -> dict:
        state = _coerce_state(state_dict)
        if state.error:
            return state.model_dump()
        return fn(state).model_dump()
    return wrapped


def _route_after_critic(state_dict: AgentState | dict) -> Literal["retry", "approve", "end"]:
    state = _coerce_state(state_dict)
    if state.error:
        return "end"
    return "retry" if state.should_retry else "approve"


def _route_after_node(state_dict: AgentState | dict) -> Literal["continue", "end"]:
    state = _coerce_state(state_dict)
    return "end" if state.error else "continue"


def build_graph():
    graph = StateGraph(AgentState)

    # nodes 
    graph.add_node("memory_retrieve", _wrap(memory_retrieve_agent))
    graph.add_node("planner",         _wrap(planner_agent))
    graph.add_node("analyst",         _wrap(analyst_agent))
    graph.add_node("selector",        _wrap(selector_agent))
    graph.add_node("trainer",         _wrap(trainer_agent))
    graph.add_node("evaluator",       _wrap(evaluator_agent))
    graph.add_node("critic",          _wrap(critic_agent))
    graph.add_node("memory_store",    _wrap(memory_store_agent))
    graph.add_node("reporter",        _wrap(reporter_agent))

    # entry 
    graph.set_entry_point("memory_retrieve")

    # fixed edges 
    graph.add_conditional_edges(
        "memory_retrieve",
        _route_after_node,
        {"continue": "planner", "end": END},
    )
    graph.add_conditional_edges(
        "planner",
        _route_after_node,
        {"continue": "analyst", "end": END},
    )
    graph.add_conditional_edges(
        "selector",
        _route_after_node,
        {"continue": "trainer", "end": END},
    )
    graph.add_conditional_edges(
        "memory_store",
        _route_after_node,
        {"continue": "reporter", "end": END},
    )
    graph.add_edge("reporter",        END)

    # conditional edges 
    graph.add_conditional_edges(
        "analyst",
        _route_after_node,
        {"continue": "selector", "end": END},
    )
    graph.add_conditional_edges(
        "trainer",
        _route_after_node,
        {"continue": "evaluator", "end": END},
    )
    graph.add_conditional_edges(
        "evaluator",
        _route_after_node,
        {"continue": "critic", "end": END},
    )
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"retry": "trainer", "approve": "memory_store", "end": END},
    )

    return graph.compile()

def _fallback_architecture_mermaid() -> str:
    return """flowchart TD
    __start__((__start__)) --> memory_retrieve[memory_retrieve]
    memory_retrieve -->|continue| planner[planner]
    memory_retrieve -->|end| __end__((__end__))
    planner -->|continue| analyst[analyst]
    planner -->|end| __end__
    analyst -->|continue| selector[selector]
    analyst -->|end| __end__
    selector -->|continue| trainer[trainer]
    selector -->|end| __end__
    trainer -->|continue| evaluator[evaluator]
    trainer -->|end| __end__
    evaluator -->|continue| critic[critic]
    evaluator -->|end| __end__
    critic -->|retry| trainer
    critic -->|approve| memory_store[memory_store]
    critic -->|end| __end__
    memory_store -->|continue| reporter[reporter]
    memory_store -->|end| __end__
    reporter --> __end__"""


def get_graph_mermaid() -> str:
    try:
        return build_graph().get_graph().draw_mermaid()
    except Exception:
        return _fallback_architecture_mermaid()

"""
def save_graph_on_startup() -> None:
    os.makedirs("artifacts", exist_ok=True)
    mermaid_str = get_graph_mermaid()

    mmd_path = os.path.join("artifacts", "pipeline_graph.mmd")
    with open(mmd_path, "w") as f:
        f.write(mermaid_str)

    import subprocess
    result = subprocess.run(
        ["mmdc", "-i", mmd_path, "-o", GRAPH_SAVE_PATH, "-b", "white"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Pipeline graph saved → {GRAPH_SAVE_PATH}")
    else:
        md_path = GRAPH_SAVE_PATH.replace(".png", ".md")
        with open(md_path, "w") as f:
            f.write(f"```mermaid\n{mermaid_str}\n```")
        print(f"mmdc not found. Mermaid saved → {md_path}")
        print("Install: npm install -g @mermaid-js/mermaid-cli")
        

def save_graph_image(path: str = "pipeline_graph.png") -> None:
    
    Save the compiled LangGraph architecture as a PNG image.
    Requires Mermaid rendering support from LangGraph or Mermaid CLI.
    
    try:
        app = build_graph()
        graph_image = app.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(graph_image)
        print(f"Graph saved to {path}")
    except Exception as exc:
        print(f"Could not save graph image with LangGraph renderer: {exc}")
        mmd_path = path.replace(".png", ".mmd")
        with open(mmd_path, "w") as f:
            f.write(get_graph_mermaid())
        print(f"Mermaid graph saved to {mmd_path}")

        """

def run_pipeline(
    dataset_path: str,
    target_column: str,
    dataset_name: str = "dataset",
    experiment_name: str = "agentic_automl",
    max_retries: int = 1,
    progress_callback=None,
) -> AgentState:
    state = AgentState(
        dataset_info=DatasetInfo(
            path=dataset_path,
            name=dataset_name,
            target_column=target_column,
        ),
        experiment_name=experiment_name,
        max_retries=max_retries,
    )
    app = build_graph()
    final_state = state.model_dump()

    for event in app.stream(state.model_dump()):
        for node_name, node_state in event.items():
            final_state = node_state
            if progress_callback:
                progress_callback(node_name, AgentState(**node_state))

    return AgentState(**final_state)
