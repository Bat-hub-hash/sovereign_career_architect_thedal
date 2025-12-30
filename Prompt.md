The Sovereign Career Architect: A 
Strategic and Technical Blueprint for 
Agentic Dominance 
1. Strategic Deconstruction of the AI-VERSE 
Competitive Landscape 
The contemporary hackathon environment, particularly one as multifaceted as AI-VERSE 
2026, functions not merely as a test of coding velocity but as a rigorous simulation of 
product-market fit and architectural foresight. To identify the singular winning concept, one 
must first perform a forensic analysis of the competing verticals—Generative AI, Agentic AI, 
and AIoT—weighing their inherent technical constraints against the judging criteria’s emphasis 
on innovation, feasibility, and impact. This report posits that while the Generative AI and AIoT 
tracks offer viable paths to completion, they suffer from "implementation saturation" and 
"simulation friction," respectively. The Agentic AI vertical, by contrast, represents the current 
frontier of artificial intelligence, offering the highest ceiling for technical differentiation and 
the most compelling narrative for "going beyond" the prompt.1 
1.1 The "Innovation vs. Saturation" Matrix in Track Analysis 
The Generative AI track, calling for a "Multilingual RAG-based Startup Funding Intelligence 
System" 1, essentially requests a specialized search engine. While the integration of Indic 
languages via models like Sarvam-1 adds a layer of localization 1, the underlying 
architecture—Retrieval-Augmented Generation (RAG)—has become the "Hello World" of the 
Large Language Model (LLM) era. RAG pipelines, which ingest PDFs and retrieve chunks 
based on vector similarity, are now commoditized workflows available via off-the-shelf 
libraries like LlamaIndex and LangChain.2 A submission in this track faces the "Utility Trap": it 
creates value, but lacks the "magic" required to win a high-stakes competition because the 
interaction model (User Query $\rightarrow$ System Retrieval $\rightarrow$ Text Output) 
remains fundamentally passive. 
The AIoT (Artificial Intelligence of Things) track, focused on "Edge Guardian" systems, 
introduces a different challenge: "Simulation Friction." The requirement to use the Wokwi 
simulator rather than physical hardware 1 creates a barrier to demonstrating visceral impact. 
While simulating WiFi RSSI trilateration for indoor localization is mathematically rigorous 1, 
visualizing invisible data packets moving between virtual ESP32 chips on a browser screen 
lacks the kinetic dynamism of a functioning product. Furthermore, the complexity of 
demonstrating decentralized mesh intelligence in a simulated environment often leads to 
demos that feel abstract to non-technical judges.3 
This leaves Agentic AI as the optimal vertical. The problem statement explicitly demands a 
departure from the "traditional chatbot," calling for a system that can "think, plan, act, and 
improve".1 This definition aligns with the emerging field of Cognitive Architectures, where 
the AI is not just a predictor of the next token but a controller of workflows and tools. The 
Agentic track allows for the deployment of a system that possesses Agency (the ability to 
initiate action), Persistence (long-term memory), and Tool Use (manipulating the external 
world). This shifts the paradigm from "Passive Information Retrieval" (Track 1) to "Active 
Autonomous Execution" (Track 2). 
1.2 The Winning Concept: The Sovereign Career Architect 
The proposed single winning idea is the "Sovereign Career Architect." This is not a chatbot 
that gives advice; it is an autonomous, stateful, voice-enabled proxy that navigates the user's 
career lifecycle. 
Core Value Proposition: 
The Architect addresses the "scattered platforms" and "manual effort" identified in the 
problem statement 1 by automating the labor of the job hunt. It autonomously navigates job 
portals using computer vision, manages application workflows, and conducts real-time, 
voice-based interview simulations in the user's native Indic language. 
Technical Differentiators: 
1. Cognitive Architecture: Built on LangGraph, the system utilizes a cyclic state machine 
to reason, plan, and self-correct, moving beyond brittle linear chains.4 
2. Persistent Memory: Leveraging Mem0, it maintains a multi-scoped memory (User, 
Session, Agent) that evolves over months, enabling the "long-term mentor" relationship.5 
3. Autonomous Action: Utilizing Browser-use, the agent breaks out of the chat window to 
interact with the web directly, clicking buttons and filling forms on sites like LinkedIn or Y 
Combinator.6 
4. Sovereign Interface: Integrating Vapi.ai for ultra-low latency voice 7 and Sarvam-1 for 
high-fidelity Indic language processing 8, it offers a "Sovereign" experience that respects 
the linguistic diversity of the Indian user base. 
The following sections detail the architectural implementation of this winning concept, 
providing a comprehensive roadmap for technical dominance. 
2. The Cognitive Core: From Linear Chains to Cyclic 
Graphs with LangGraph 
To fulfill the mandate of a system that can "think, plan, act, and improve" 1, one must abandon 
the linear "Chain" architecture common in early LLM applications. Linear chains (Input 
$\rightarrow$ Prompt $\rightarrow$ Output) are brittle; if a tool fails or a plan is deemed 
suboptimal, the execution collapses. The winning solution necessitates a Cyclic State 
Machine. 
2.1 The Shift to Stateful Graphs 
LangGraph is the industry-standard framework for building stateful, multi-actor 
applications.9 Unlike linear chains, LangGraph models the agent's logic as a directed graph 
where nodes represent computational steps (Reasoning, Tool Execution, Memory Update) and 
edges represent the control flow. This allows for Reflective Loops: the agent can generate a 
resume, critique it against the job description, and loop back to the generation node to refine 
it before presenting it to the user. 
The architecture of the Sovereign Career Architect is defined by a global state object, 
AgentState, which persists across the graph's execution. This state is the "short-term working 
memory" of the agent, accumulating context as it moves through the graph nodes. 
Table 1: The AgentState Schema Definition 
State Key Data Type Description and Function 
messages List The appended history of all 
interactions in the current 
session. Used for 
maintaining conversational 
context. 
user_profile Dict[str, Any] Structured data retrieved 
from Mem0 (e.g., Skills: 
["Python", "LangGraph"], 
Goal: "Remote Dev"). 
current_plan List[str] The Directed Acyclic Graph 
(DAG) of tasks generated 
by the Planner Node (e.g., 
1. Search Jobs, 2. Filter, 3. 
Apply). 
tool_outputs List Raw data captured by the 
Executor Node (e.g., 
scraped job descriptions, 
interview feedback). 
critique 
str 
The output of the Reviewer 
Node, containing specific 
feedback on the agent's 
own performance. 
retry_count 
int 
2.2 Node Architecture and Control Flow 
A counter to prevent 
infinite loops during error 
recovery (e.g., if a website 
is down). 
The graph is composed of five primary nodes, each encapsulating a distinct cognitive 
function. The edges between these nodes are conditional, determined by the Supervisor 
logic. 
1. The Profiler Node (Contextualization): This node is the entry point. It ingests the user's 
raw input and queries the Mem0 layer to retrieve relevant long-term context.5 If the user 
says, "Find me a job," the Profiler retrieves the specific preferences (salary, location, role) 
stored in previous sessions, enriching the prompt before it reaches the reasoning model. 
2. The Planner Node (Reasoning): Utilizing a high-reasoning model (e.g., Llama-3-70B via 
Groq), this node analyzes the enriched context and generates a step-by-step plan. It acts 
as the "Prefrontal Cortex," deciding what needs to be done without worrying about how.11 
3. The Executor Node (Action): This is the "Motor Cortex." It interfaces with external tools, 
primarily the Browser-use library for web interaction and search APIs. It executes the 
specific steps defined by the Planner.4 
4. The Reviewer Node (Reflection): This node acts as a quality gate. It compares the 
tool_outputs against the user_profile and current_plan. If the scraped jobs do not match 
the user's skills, the Reviewer rejects the output and routes the flow back to the Planner 
with specific feedback (e.g., "Search query too broad, refine keywords"), fulfilling the 
"Self-Correction" requirement.1 
5. The Archivist Node (Consolidation): At the end of a successful execution, this node 
parses the interaction for new salient facts (e.g., "User learned a new framework") and 
pushes them into Mem0 for long-term storage, ensuring the agent "improves over time".5 
2.3 Implementing the "Go Beyond" Logic 
The LangGraph architecture allows for sophisticated "Human-in-the-Loop" (HITL) workflows, 
a key differentiator for "Responsible AI." Before the Executor Node performs a high-stakes 
action—such as submitting a job application—the graph can be configured to pause 
execution and wait for user approval.12 This is implemented using LangGraph's 
interrupt_before functionality. The agent presents a summary ("I am about to apply to Google 
with Resume V2") and halts. Only upon receiving a confirmation signal does the state 
transition continue. This feature demonstrates a nuanced understanding of agency—it is 
autonomous but accountable. 
3. The Hippocampus: Persistent and Evolving Memory 
with Mem0 
A critical failure mode of standard LLM agents is "catastrophic amnesia"—the resetting of 
context after every session. The problem statement explicitly demands a "persistent career 
memory that updates as the user grows".1 A simple vector database (like Pinecone) is 
insufficient because it lacks semantic organization and requires manual management of 
embeddings. The winning solution integrates Mem0 (formerly EmbedChain), a specialized 
memory layer designed for personalized AI interactions.5 
3.1 The Multi-Scope Memory Model 
Mem0 differentiates itself by organizing memory into hierarchical scopes, which is essential 
for mimicking a human mentor-mentee relationship. The Sovereign Career Architect utilizes 
three distinct memory scopes to manage the complexity of a user's career trajectory.13 
Table 2: Mem0 Memory Scopes and Implementation Strategy 
Memory Scope 
User Memory 
Session Memory 
Function within Career 
Architect 
Stores static and slowly 
evolving facts. Examples: 
"Education: B.Tech CS," 
"Preferred Stack: MERN," 
"Aversion: Night Shifts." 
Implementation Detail 
Persists globally across all 
sessions. Accessed via a 
unique user_id hash. This 
memory is queried at the 
start of every session to 
"prime" the agent. 
Tracks the immediate 
context of the current task. 
Examples: "Currently 
viewing Job ID #992," 
Temporary storage that 
provides short-term 
continuity. It allows the 
user to say "Change the 
second paragraph" without 
"Drafting Cover Letter V3." 
Agent Memory 
restating which document 
is being discussed. 
Stores the agent's learned 
strategies for the user. 
Examples: "User responds 
better to critical feedback," 
"User prefers lists over 
paragraphs." 
3.2 Dynamic Context Injection 
Meta-memory used to 
dynamically tune the 
system prompt. This allows 
the agent to adapt its 
personality to the user, not 
just its content. 
The integration of Mem0 transforms the agent's interaction model. When a user initializes a 
chat, the system does not start with a blank slate. The Profiler Node executes a 
mem0.search(query, user_id) call.14 
Consider a scenario where a user returns after two weeks and asks, "Any updates on those 
React roles?" 
1. Naive Agent: "Which React roles are you referring to?" (Fails the "Long-term mentor" 
criteria). 
2. Mem0 Agent: The system retrieves the vector embedding for "React roles" associated 
with the user_id. It finds memories tagged with "Date: 2 weeks ago" and "Context: 
Bangalore Startups." 
3. Synthesized Response: "I've continued tracking the Bangalore React openings we 
discussed. Two new positions at Series B startups have opened up that match your 
salary expectations." 
This capability is achieved through Mem0's combination of vector search (for semantic 
similarity) and graph databases (for relationship mapping).15 The graph component allows the 
system to infer relationships: if the user updates their skill in "Next.js," Mem0 updates the 
graph node connecting "User" to "Frontend Engineering," automatically adjusting future job 
recommendations without explicit instruction. 
3.3 The Feedback Loop 
The "Archivist Node" ensures that the memory is not static. After every significant 
interaction—such as a mock interview or a job application—the agent extracts "salient facts." 
For instance, if a user fails a mock interview question about "System Design," the Archivist 
pushes a new memory: "User has a knowledge gap in Distributed Systems." 
In the next session, the Planner Node will see this memory and proactively suggest: "Before 
we apply to this Senior role, should we review some Distributed Systems concepts?" This 
proactive behavior is the hallmark of a true "Agentic" system. 
4. The Hands: Autonomous Action via Browser-use 
The most visually arresting differentiator for this project is the ability to act. Most career bots 
simply output text: "You should apply to Google." The Sovereign Career Architect uses the 
Browser-use library to actually perform the application process.6 This library allows an LLM 
to control a headless Chromium instance via the Chrome DevTools Protocol (CDP), translating 
natural language instructions into DOM interactions.16 
4.1 Computer Vision and DOM Analysis 
Standard scraping tools (like BeautifulSoup) fail on modern, dynamic web applications (SPAs) 
built with React or Vue because they rely on brittle CSS selectors. Browser-use overcomes 
this by using a multimodal approach. It injects a script that highlights interactive elements on 
a webpage with unique numeric IDs (e.g., , ). It then captures a screenshot and the simplified 
DOM structure.17 
This multimodal input (Visual + Text) is sent to a Vision-Language Model (VLM) like GPT-4o. 
The VLM "sees" the page exactly as a human does. It identifies the "Apply Now" button not 
because it has a specific div id, but because it visually resembles a primary action button in 
the context of the page layout. 
The Application Workflow: 
1. Navigation: The Executor Node receives a command: "Apply to the Senior Engineer role 
at Anthropic." It calls agent.goto(url). 
2. Perception: The VLM analyzes the screenshot. It identifies the "Apply" button as element 
``. 
3. Action: The agent executes agent.click(element_id=42). 
4. Form Filling: The agent encounters a form. It queries Mem0 for the user's "LinkedIn 
URL" and "Portfolio." It then executes agent.type(element_id=55, 
text="https://linkedin.com/in/user").18 
5. Handling Complexity: If the site requires a resume upload, the agent locates the file 
input, interacts with the operating system's file dialog (simulated via Playwright), and 
uploads the specific PDF generated by the Resume Builder module. 
4.2 Handling Instability and Anti-Bot Measures 
Browser automation is notoriously brittle. Anti-bot systems (like Cloudflare) often flag 
headless browsers. To ensure the reliability required for a hackathon demo: 
● Stealth Mode: The solution utilizes modified browser headers and user-agent strings to 
mimic legitimate human behavior. The Browser-use library supports these 
configurations via its underlying Playwright architecture.19 
● Vision-Based Recovery: Unlike selector-based automation, if the DOM structure 
changes (e.g., the website updates its CSS), the VLM can still "see" the button. This 
makes the agent robust to frontend updates.20 
● Rate Limiting: The Executor Node implements a "human-like" delay between actions 
(e.g., 500ms - 1500ms) to avoid triggering rate limiters. 
This capability transforms the system from a "Career Advisor" to a "Career Agent." It actively 
reduces the manual toil of the job hunt, directly addressing the user pain point of "constant 
manual effort".1 
5. The Voice: Sovereignty and Real-Time Interaction 
with Vapi.ai and Sarvam-1 
To secure the "Go Beyond" points and address the "Indic Language Support" requirement 1, 
the Architect sheds the keyboard interface for a real-time voice conversation. This transforms 
the experience from "filling a form" to "talking to a mentor." 
5.1 The Interface of the Future: Vapi.ai 
Vapi.ai is chosen for its ability to handle the entire voice orchestration layer with ultra-low 
latency (sub-500ms).7 Vapi manages the complex pipeline of Voice Activity Detection (VAD), 
Speech-to-Text (STP), and Text-to-Speech (TTS). 
Crucial Feature: Server-Side Tool Calling 
Vapi is not just a voice bot; it is a gateway to the agent's logic. Vapi can be configured with 
"Tools" that map to the LangGraph nodes. When the user says, "Schedule a mock interview," 
Vapi's orchestration layer recognizes this intent and triggers a function call to the backend API 
hosting the LangGraph agent.21 This seamless integration ensures that the voice interface is 
not a gimmick but a fully functional command center for the agent. 
5.2 Sovereign Intelligence: Sarvam-1 
While Vapi handles the plumbing, the intelligence must be culturally resonant. The system 
integrates Sarvam-1, India's first sovereign 2-billion parameter LLM, to handle Indic language 
interactions.8 
Why Sarvam-1? 
Sarvam-1 is optimized for 10 Indic languages (Hindi, Tamil, Telugu, etc.) and trained on a 
high-quality corpus of 2 trillion tokens.8 Its tokenizer is 2-4x more efficient for Indic scripts 
than Llama-3 or GPT-4. Standard models suffer from "high token fertility" in Indic 
languages—a single Hindi word might be broken into 4-5 tokens, increasing latency and cost. 
Sarvam-1 achieves a fertility rate of ~1.4-2.1, making it the most efficient model for this use 
case.8 
The Indigenized Interview Workflow: 
1. User Input: "Mujhe React interview ki tayari karni hai" (I want to prepare for a React 
interview). 
2. Transcription: Vapi (using Deepgram with Hindi support) transcribes the audio. 
3. Routing: The LangGraph Supervisor Node identifies the language as Hindi. 
4. Sovereign Execution: The request is routed to a specialized node running Sarvam-1 (via 
Hugging Face Inference Endpoints). 
5. Generation: Sarvam-1 generates a technical interview question in natural, conversational 
Hindi: "React mein useEffect hook ka kya kaam hai?" (What is the function of the 
useEffect hook in React?). 
6. Synthesis: Vapi receives this text and synthesizes it using an Indian-accented voice 
model (e.g., from ElevenLabs or PlayHT), providing a culturally immersive experience. 
This integration proves to the judges that the team is deeply engaging with the Indian AI 
ecosystem, fulfilling the spirit of "Sovereignty" and "Inclusivity" championed by the AI-VERSE 
theme. 
6. Implementation Roadmap: The 48-Hour Sprint 
Success in a hackathon is a function of scope management. The following roadmap breaks 
down the development of the Sovereign Career Architect into four 12-hour phases, ensuring a 
functional deliverable at every milestone. 
6.1 Phase 1: The Core Graph (Hours 0-12) 
Objective: Establish the LangGraph backbone and state management. 
● Tasks: 
○ Define the AgentState schema (TypedDict). 
○ Implement the Planner, Executor, and Reviewer nodes using basic LangChain 
prompts. 
○ Create the Supervisor logic to route between nodes. 
○ Deliverable: A text-based CLI agent that can take a query ("Plan my career in 
DevOps") and output a sequence of logical steps. 
6.2 Phase 2: Memory Integration (Hours 12-24) 
Objective: Give the agent persistent memory. 
● Tasks: 
○ Set up a hosted Qdrant or Chroma vector store. 
○ Initialize the Mem0 client and configure User/Session scopes. 
○ Implement the Profiler node to fetch memories at the start of a session. 
○ Implement the Archivist node to save memories at the end. 
○ Deliverable: A demo where the agent "remembers" the user's name, skills, and goals 
after a server restart. 
6.3 Phase 3: The "Hands" (Hours 24-36) 
Objective: Implement autonomous web interaction. 
● Tasks: 
○ Install Browser-use and Playwright dependencies. 
○ Build specific tool definitions for "Job Search" (e.g., navigating to LinkedIn/YC Jobs). 
○ Implement the "Stealth Mode" configurations to avoid bot detection. 
○ Deliverable: A video recording of the agent autonomously finding a job posting, 
extracting the requirements, and saving them to the state. 
6.4 Phase 4: The "Voice" & Polish (Hours 36-48) 
Objective: Integrate Vapi.ai and Sarvam-1 for the "Wow" factor. 
● Tasks: 
○ Configure a Vapi.ai assistant with the "Function Calling" enabled. 
○ Set up the Sarvam-1 inference endpoint on Hugging Face or use the API. 
○ Connect the Vapi webhook to the LangGraph API (exposed via FastAPI/Flask). 
○ Deliverable: A live demo of a user talking to the agent in Hindi, practicing for an 
interview, and having the agent apply for a job. 
7. Deep Dive: Technical Challenges and Mitigation 
7.1 Challenge: Hallucination in Career Advice 
Risk: The agent might recommend non-existent certifications or obsolete technologies (e.g., 
"Get the Google Cloud Certified Associate 2019"). 
Solution: The Reviewer Node implements a "Fact-Check" step. It uses the Browser-use tool to 
perform a quick Google Search verification for any specific claim. If the search returns 
negative results or indicates obsolescence, the advice is flagged and regenerated. This 
Grounded Generation approach ensures reliability. 
7.2 Challenge: Infinite Loops in Browser Automation 
Risk: The agent gets stuck trying to close a popup ad or finding a button that doesn't exist.22 
Solution: 
1. Maximum Retry Budget: The Executor Node has a strict counter. If an action fails 3 
times, it aborts and notifies the Planner. 
2. Vision-Based Recovery: Utilizing the VLM capabilities of Browser-use allows the agent 
to "see" the "X" button to close a popup, even if the element ID is obfuscated. 
3. Human Handoff: If the agent is truly stuck, it triggers an interrupt state, alerting the user 
via the UI to manually intercede (e.g., solve a CAPTCHA) before resuming control. 
7.3 Challenge: Latency in Voice Interaction 
Risk: If the RAG/Reasoning pipeline takes 5+ seconds, the voice conversation feels broken and 
unnatural. 
Solution: 
1. Streaming Responses: Vapi.ai supports streaming. The LLM output is piped directly to 
the TTS engine token-by-token, reducing the "Time to First Byte" (TTFB). 
2. Optimistic Acknowledgment: The architecture uses "Filler Phrases." As soon as the 
user stops speaking, the agent immediately triggers a low-latency filler ("That's a great 
question, let me think..."), masking the processing time of the heavier Sarvam-1 or 
Planner model. 
8. Impact Assessment: Defining the "Go Beyond" 
The problem statement encourages participants to "Go Beyond." The Sovereign Career 
Architect achieves this through three specific vectors of impact: 
1. Psychological Safety: Job hunting is stressful. By using an AI for interview practice, the 
user fails safely. The Vapi-enabled voice mode creates a low-stakes environment to 
practice high-stakes conversations. The agent can analyze tone, pitch, and hesitation 
(using Vapi's metadata) to provide behavioral feedback, not just content feedback. 
2. Democratization of Mentorship: By supporting Indic languages via Sarvam-1, the 
solution democratizes access to high-quality career mentorship. A student in a Tier-3 
college who is technically sound but struggles with English can now practice interviews in 
Hinglish, bridging the confidence gap. This directly addresses the "Equity" component of 
Responsible AI. 
3. Agency and Autonomy: The shift from "searching for jobs" to "having jobs found for 
you" fundamentally changes the user's relationship with the platform. It reduces the 
cognitive load of the job hunt, which is often the biggest barrier to career progression for 
under-resourced candidates. 
9. Conclusion 
The Sovereign Career Architect is not merely a collection of API calls; it is a cohesive 
Cognitive System designed to win. It aligns perfectly with the judging criteria by 
demonstrating: 
● Technical Complexity: Through the use of LangGraph's cyclic state machine and 
Mem0's persistent memory architecture. 
● Innovation: By deploying Browser-use for "Agentic Action" rather than passive retrieval. 
● Cultural Relevance: By integrating Sarvam-1 for sovereign, multilingual support. 
● User Experience: By utilizing Vapi.ai for a seamless, hands-free voice interface. 
By focusing on the Agentic AI track, the solution avoids the "simulation trap" of AIoT and the 
"commodity trap" of basic GenAI. It is built on robust, production-ready libraries yet pushes 
the boundaries of what is expected in a hackathon prototype. It is technically rigorous, 
socially impactful, and strategically designed to secure victory. 
Detailed Technical Appendix: 
Implementation Specs 
A.1 LangGraph Node Logic (Pseudo-code) 
The following pseudo-code illustrates the logic within the Supervisor or Router node, which 
dictates the flow of the agent. 
Python 
# State Definition 
class AgentState(TypedDict): 
messages: Annotated, operator.add] 
next_step: str 
user_profile: Dict[str, Any] 
# The Router Node 
def supervisor_node(state: AgentState): 
last_message = state['messages'][-1] 
# Use LLM to classify intent 
classification = llm.invoke( 
) 
f"Classify intent: {last_message.content}. Options:" 
if classification == "SEARCH_JOB": 
return {"next_step": "browser_agent"} 
elif classification == "INTERVIEW_PREP": 
return {"next_step": "voice_interviewer"} 
else: 
return {"next_step": "general_chat"} 
# The Graph Construction 
workflow = StateGraph(AgentState) 
workflow.add_node("supervisor", supervisor_node) 
workflow.add_node("browser_agent", browser_search_node) 
workflow.add_node("voice_interviewer", sarvam_interview_node) 
workflow.add_conditional_edges("supervisor", lambda x: x["next_step"]) 
A.2 Mem0 Integration Logic 
This snippet demonstrates how Mem0 is used to fetch context before the LLM generates a 
response, ensuring continuity. 
Python 
from mem0 import MemoryClient 
client = MemoryClient(api_key="...") 
def retrieval_node(state): 
user_id = state["user_id"] 
query = state["messages"][-1].content 
# Semantic search for relevant past memories 
memories = client.search(query, user_id=user_id) 
# Format memories into a system prompt string 
context_str = "\n".join([m['text'] for m in memories]) 
# Inject into state for the next node 
return {"memory_context": context_str} 
A.3 Browser-use Application Logic 
This illustrates the high-level command structure used to drive the headless browser via the 
Executor Node. 
Python 
from browser_use import Agent 
async def apply_to_job(job_url, user_data): 
# The 'task' string is parsed by the LLM into specific browser actions 
agent = Agent( 
task=f"Go to {job_url}. Click 'Apply'. Fill name with {user_data['name']}. Upload resume from local 
path. Do not submit, just pause.", 
llm=gpt4o_model 
) 
history = await agent.run() 
return history 
A.4 Vapi.ai + Sarvam-1 Configuration 
The conceptual configuration for routing Vapi voice data to a custom Sarvam-1 server. 
JSON 
{ 
"transcriber": { 
"provider": "deepgram", 
"language": "hi" // Hindi support 
}, 
"model": { 
"provider": "custom-llm", 
"url": "https://my-sarvam-inference-endpoint.com/v1/chat/completions", 
"systemPrompt": "You are a kind but strict interviewer. Speak in Hindi." 
}, 
"voice": { 
"provider": "11labs", 
"voiceId": "indian_accent_male" 
} 
} 
Works cited 
1. Problem Statement.pdf 
2. Sarvam AI | Sovereign Indian AI Ecosystem for LLMs, Agents, and AI Assistants, 
accessed December 26, 2025, https://www.sarvam.ai/ 
3. Mesh size limits - ESP32 Forum, accessed December 26, 2025, 
https://esp32.com/viewtopic.php?t=5919 
4. Building a Stateful AI Trading Agent with LangGraph | by Rejith Retnan - 
DataDrivenInvestor, accessed December 26, 2025, 
https://medium.datadriveninvestor.com/building-a-stateful-ai-trading-agent-with-langgraph-a31521bc14ea 
5. Build Smarter AI Agents: Mem0 + LangGraph Guide - DataCouch, accessed 
December 26, 2025, 
https://datacouch.io/blog/build-smarter-ai-agents-mem0-langgraph-guide/ 
6. Browser Use | Technology Radar | Thoughtworks United States, accessed 
December 26, 2025, 
https://www.thoughtworks.com/en-us/radar/languages-and-frameworks/browser-use 
7. How to Build a Smart AI Voice Assistant with Vapi - Analytics Vidhya, accessed 
December 26, 2025, 
https://www.analyticsvidhya.com/blog/2025/11/vapi-ai-voice-assistant/ 
8. Sarvam 1, accessed December 26, 2025, https://www.sarvam.ai/blogs/sarvam-1 
9. LangGraph: Build Stateful AI Agents in Python, accessed December 26, 2025, 
https://realpython.com/langgraph-python/ 
10. langchain-ai/open_deep_research - GitHub, accessed December 26, 2025, 
https://github.com/langchain-ai/open_deep_research 
11. Build AI Agents with browser-use and Scraping Browser - Bright Data, accessed 
December 26, 2025, 
https://brightdata.com/blog/ai/browser-use-with-scraping-browser 
12. Mem0 Tutorial: Persistent Memory Layer for AI Applications - DataCamp, 
accessed December 26, 2025, https://www.datacamp.com/tutorial/mem0-tutorial 
13. LangGraph - Mem0 Documentation, accessed December 26, 2025, 
https://docs.mem0.ai/integrations/langgraph 
14. Implement Long-Term Memory in Your AI Agents with Mem0, Azure AI Foundry 
and AI Search | by Eitan Sela - Medium, accessed December 26, 2025, 
https://medium.com/microsoftazure/implement-long-term-memory-in-your-ai-a
gents-with-mem0-azure-ai-foundry-and-ai-search-56efd8683c03 
15. Browser Use: Technical Expert Review & Tests - GoLogin, accessed December 26, 
2025, https://gologin.com/blog/browser-use-technical-expert-review-tests/ 
16. Browser-Use: Open-Source AI Agent For Web Automation - Labelerr, accessed 
December 26, 2025, https://www.labelerr.com/blog/browser-use-agent/ 
17. Double Your Efficiency with AI + Browser-use! | by Meng Li | Top Python Libraries | 
Medium, accessed December 26, 2025, 
https://medium.com/top-python-libraries/double-your-efficiency-with-ai-browse
r-use-f2656be7a6b0 
18. Browser Use - Enable AI to automate the web, accessed December 26, 2025, 
https://browser-use.com/ 
19. Browser Use vs Hyperbrowser AI: Which is Better? (November 2025) - Skyvern, 
accessed December 26, 2025, 
https://www.skyvern.com/blog/browser-use-vs-hyperbrowser-ai/ 
20. Vapi - Build Advanced Voice AI Agents, accessed December 26, 2025, 
https://vapi.ai/ 
21. browser-use sucks !! : r/AI_Agents - Reddit, accessed December 26, 2025, 
https://www.reddit.com/r/AI_Agents/comments/1hzt9tt/browseruse_sucks/ 