# Raven: Reasoning Agents via Explanatory laNdmarks

This is an implementation of the RAVEN approach. As reference application, this code implements a Finance analysis engine that can carry out complex analytic tasks.


*Features* of Agent

- Reflect uses OpenAI API 
- Act Uses yfinance for ticker data and local functions that implement basic quantitative analyses: mean, volatility, ratios, momentum price, and symbol lookup
- State Space object has the parameters of the State Space.
- Value uses simpler local LLMs to ascertain the validity of the actions
- Enhance is implemented an additional OpenAI step

State of the world is kept and managed in a domain specific Object. Currently consists of span, and ticker list.

Next to do: Make it Mixed Initiative so that it can prompt user for inputs and wait.

## Evaluation Tasks

Here are some notes on the 2 key applications:

1. Financial Analyst: Pulls data, runs exploratory data analysis, provides insights,  builds a hypothesis, backtests the hypothesis.

2. Timed Chess: Gets the move from oponent, thinks and makes a move. Thinks while the oponent is thinking.
    https://github.com/LeelaChessZero/lc0/wiki/Getting-Started





## About Asynchronous Agents

There are several frameworks specifically designed for asynchronous collaboration of AI agents. Here are the main ones:
Top Frameworks for Asynchronous Agent Collaboration
1. AutoGen / AG2 (Microsoft)
AutoGen allows multiple agents to communicate by passing messages in a loop with asynchronous agent collaboration FREDHealth Affairs. Agents communicate through asynchronous messages, allowing them to work independently and in parallel, enhancing system scalability and responsiveness Health Affairs. With AG2's event-driven architecture, agents can work concurrently rather than sequentially, reducing bottlenecks in multi-step workflows OECD. It's built on the actor model, making it particularly suitable for distributed systems.
2. LangGraph
LangGraph allows you to model multiple agents as individual nodes or groups, each with its own logic, memory, and role in the system FRED. The graph-based architecture enables conditional logic and multi-team coordination, making it suitable for complex asynchronous workflows where agents maintain their own state.
3. CrewAI
CrewAI is built around collaboration where you define the crew structure, assign roles, wire in tools, and let the agents coordinate their work through structured messages, task handoffs, and feedback loops OECD. CrewAI offers two modes: self-organizing crews (where agents determine their own collaboration patterns) and explicit CrewAI Flows (where you script exact interactions) OECD.
4. OpenAI Swarm
Swarm enables multiple AI agents to collaborate in real time with asynchronous and structured communication where agents can post messages, read each other's outputs, and respond OECD. However, it's still experimental and primarily for research and education rather than production use.
5. LangChain
LangChain supports both synchronous and asynchronous workflows, making it suitable for production-grade pipelines III, though it's more focused on single-agent orchestration with tools rather than multi-agent coordination.
Key Features of Asynchronous Collaboration
Common patterns across these frameworks include:

Message passing systems that allow agents to communicate without blocking
Shared memory banks where agents write/read to shared memory banks, allowing asynchronous coordination for task handoff and state sharing Health Affairs
Event-driven architectures that enable concurrent agent operation
Role-based organization where specialized agents handle different aspects of tasks
Human-in-the-loop capabilities for oversight and intervention when needed

Which Framework to Choose?

For production-ready asynchronous systems: AutoGen/AG2 is the most mature
For graph-based workflows with complex state management: LangGraph
For team-based collaboration with intuitive abstractions: CrewAI
For research and experimentation: OpenAI Swarm

The field is evolving rapidly, with most frameworks adding or improving asynchronous capabilities as multi-agent systems become more common in production environments.


References
1. A survey on large language model based autonomous agents  https://arxiv.org/pdf/2308.11432 (Has benchmarks. Survey of domain-specific applications)
