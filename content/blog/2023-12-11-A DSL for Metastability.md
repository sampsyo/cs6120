+++
title = "A DSL for Studying Metastability"
[extra]
bio = """
  I am Ali Farahbakhsh, a 2nd-year PhD student at Cornell CS. I do research on distributed systems.
"""
latex = true
[[extra.authors]]
name = "Ali Farahbakhsh"

+++
# Intro
Researchers have recently characterized a new class of computer system failures: [metastable failures](https://www.usenix.org/conference/osdi22/presentation/huang-lexiang).
A general blueprint for these failures is as follows:
1. The system is operating within a given environment.
2. A trigger event happens. The trigger can come in different flavors: server crash, network partition, load surge, etc.
3. The trigger pushes the system into a self-sustaining loop, in which the system is also "unavailable".
4. Even after the triggering effect is remedied, the system remains in the self-sustaining loop.

I have been trying to rigorously define metastability as a property of a suitably general system abstraction. We have a working definition that seems to capture the gist of it. However, since we are interested in actual computer systems that might experience metastability, it would be great if we could:
1. express a generic "computer system", especially one that resembles the cloud in that it is made of clusters of homogenous machines,
2. and analyze whether a given system - expressed in the previous step - "is metastable or not".

The goal of this project was to design such a DSL, and to test its correctness. The desired outcome is a DSL with appropriate syntax and semantics, and an interpreter that receives a system expressed in this DSL and returns an execution trace. Such a trace can then be used to assess whether the system would suffer from metastability.

# Design
When I was first introduced to metastable failures, the main examples that were being used to study it were mostly about systems having a finite capacity to process stuff.
Perhaps the most famous example is a retry storm, which roughly proceeds as follows:

1. The system is operating under heavy load, such that it is about to be overloaded.
2. A load surge happens, pushing the system into the overloaded state. This causes the system to drop some requests, and some requests timeout.
3. The requests that are being timed out start retrying, and the retries keep growing as long as the overload state is sustained.
4. After a certain threshold, even if the load surge is mitigated and the load returns to a level that *should* be safe according to the spec, the retries keep the system in the overload state.

Based on this, my first attempt to capture this type of behavior was to imitate a method that researchers have used for other systems that demonstrate such *dynamic* behaviors: I modeled a computer system as a group of agents with finite capacity that interact according to predetermined rules.
This finitude was captured through the following abstraction for an agent:

1. An agent has a finite input queue and a finite output queue.
2. An agent has finite receiving and sending rates on its input and output queues, respectively.
3. An agent has a finite processing rate, where "processing" means taking something from the input queue and putting it into the output queue (note that there is no content to processing).
4. Optionally, an agent can have a finite pending bag: an abstraction to hold the requests that are waiting for an acknowledgement.

Under such a model, the execution of the system can be viewed as the recurrent receive-process-send cycle carried out by individual agents, which obviously requires a network abstraction to move messages around. This effort - which was carried out before the course - resulted in a simulator that helped us capture some metastability ghosts with this finite-capacity flavor.

However, once I started looking around for more examples, I gradually realized that if we generalized a bit, other behaviors from totally different contexts - still within the purview of computer systems - also belonged to metastability as I understood it.
This was what led to the realization that a DSL is more suitable than a simulator for studying metastabilty, since the execution laws are baked into the simulator while a DSL gives more freedom at expressing systems.
The basis is still the same: agents with finite capacity.
However, this time the agents also have *internal state*, with which they can play according to a set of primitive rules.
Moreover, the notion of a message is subsumed by that of a *request*.
A request is very similar to a packet: it has different fields that can take different values.

The rest of this post briefly introduces the current version of the DSL.

## Syntax
For reasons that are beyond this post, I decided to make the DSL look like [NetKAT](https://dl.acm.org/doi/pdf/10.1145/2578855.2535862) - at least syntactically for now.
I might regret this later, but here goes nothing.
This is the current version of the grammar:

Fields: $f ::= f_1\ | \ f_2\ | \dots | \ f_k$

Values: $v ::= \texttt{n}\ | \ \texttt{true}\ |\ \texttt{false}\ | \ v_1\ | \dots |\ v_m$

Requests: $r ::= \{f_1 = v_1,\ \dots,f_k = v_k\}\ | \ r_1\ | \dots | \ r_j\ | \ \texttt{nil}$

Elements: $el ::= el_1\ | \dots | \ el_i$

Queues: $q ::= \{el_1 = r_1,\dots,el_l = r_l\}$

State Variables: $a$

Predicates: $x,y ::= \texttt{true}\ |\ \texttt{false}\ |\ a == v\ |\ x\ \texttt{or}\ y\ | \ x\ \texttt{and}\ y \ | \ \neg\ x$

Actions: $\alpha, \beta ::= \texttt{nop}\ | \ a = v \ | \ a = r \ | \ a = \texttt{get}(q) \ | \ \texttt{put}(q, a) \ | \ a\ +=\ 1 \ | \ x.\alpha + (\neg x).\beta \ | \ \alpha.\beta \ | \ \alpha^*$

Some notes are in order (I will not include inference rules or stepping semantics in this post, so if one day you find yourself in dire need for a DSL to study metastability, this is all you have got):
1. Every "operation" in this DSL is an action. The meaning of the term "operation" will be clearer in the next section.
2. The "+" operator for an action designates nondeterministic choice in general, and that is the intended long-term purpose here. For now, I have limited it to deterministic branching based on the value of a predicate.
3. The "*" operator is our favorite Kleene star. The intended long-term behavior is repeated application of an action. In practice, just like branching, it will be guarded by a predicate for now (I did not want to clutter the star - it's so beautiful!).
4. $\texttt{put}$ only accepts state variables with a request for their value as its second argument. I could not find a neat way of expressing this in the grammar (yes, I'm not good with grammars).
5. An agent is assumed by default to have a single input queue and a single output queue, named $\texttt{inq}$ and $\texttt{outq}$, respectively.
6. By default, each piece of code written with this grammar is assuming that it is expressing a single agent. A separate file - a config file for instance - should establish the mapping between a set of agents and their "code".
7. I have limited my analysis to agents that follow the receive-process-send pattern. This limitation is not expressed in the grammar, and this is fine. While you can write many agents with the DSL, the theoretical analysis will not study agents that do not start with some logic for receiving requests, followed by a logic to process them, followed by some logic to send them.
8. So, what does it mean to "process a request"? You just take it from the input queue, apply some actions to your state variables according to the request, and then put a corresponding request in your output queue if required.
9. $\texttt{n}$ is an integer.

## Semantics
There are several ways of capturing the semantics of a system expressed in this DSL.
First of all, note that while the DSL expresses what happens when an agent executes itself, it does not talk about the concurrent execution semantics of a group of agents.
Therefore, the semantics for the overall execution of the system requires two parts:
1. The semantics of the DSL.
2. The concurrent execution semantics.

### DSL Semantics
Inspired from the literature on dynamical systems, I have chosen to regard every step that an agent takes as a map: a map starting from a state and ending at a state.
What is state, I hear you asking. State for an agent is its internal state variables plus the contents of its queues. 
Moreover, I have assumed for now that the agent is *deterministic*.
Therefore, one can model an agent as a function from states to states.
To make things simpler, I have also assumed that the entirety of the code for an agent is applied at once whenever it is invoked.
That is, the agent receives an input state, goes through all of the actions in its code via a series of small steps, and returns another state.
From the point of view of an external observer, all of this is just a big step.
Most of the actions in the grammar are self-explanatory.
Here are some details:

1. Predicates have no effect on the state. Only their value at the time is important. Therefore, all an interpreter has to do is to evaluate the predicate according to some simple rules, and to take the appropriate action based on the value.
2. As I also mentioned in the syntax section, the Kleene star is always guarded by a predicate for now. Therefore, the action specified by the star operation keeps being applied to the state in small-step fashion until the value of the predicate becomes $\texttt{false}$.
3. When multiple agents are applying their code to the global state, their actions commute since they are working on different parts of the global state. The concurrency related to messaging is handled in the concurrent execution semantics.

### Concurrent Execution Semantics
Once we have a number of agents and know how to interpret their code, it is time to execute the entire system.
I have assumed that the system is being executed in discrete time with its units called *steps*.
There is an entity called the "network", that at each step receives the requests sent by the agents, and delivers the requests that are due for reception.
The network can be modeled as synchronous or asynchronous, and I have opted for synchronous for now.
The concurrency is handled by the order in which the network decides to receive requests from the agents and to deliver requests to the agents.
It can reorder the requests however it wants, subject to potential constraints (e.g., FIFO between the same sender and receiver).

Note that there are entire languages and logics just to study this networking behavior.
This is, of course, bread and butter for distributed systems, but it is still subject to on-going research.
I have not tried to express these networking behaviors via existing frameworks, and I am planning to capture them with adequate infrastructure and API in the execution engine for the whole system.

There is also an entity called the environment, which is responsible for *triggers*.
The environment is not expressed by the DSL, and is assumed to operate based on pure choice: each "action" for the environment is just an arbitrary state transformation subject to the limitations imposed on the environment.
These limitations are expressed by specifying the variables that the environment can touch.
For instance, the code for some agent can be wrapped with a giant branching action with a guard $\texttt{not crashed}$, and a $\texttt{nop}$ for the else clause.
The environment can then choose to crash and uncrash the agent however it likes.

Based on these, a general execution of the system is a repeating cycle of the following:
1. The environment takes an action.
2. The system as a whole takes an action, i.e., each agent takes a big step.
3. The network handles the messages.

## Example
Time for an example.
This example is from an old paper that was trying to study a real-world phenomenon in routing networks.
We have identified the phenomenon as being metastable, which is good news for the DSL's ability to express systems in which metastability is observed.
There are other examples as well (the retry storm and a membership scenario from an industry setup), but for the sake of brevity I have not included them here.
Note that I have heavily used sugaring, as you can imagine it is not pleasant to write actual code in the specified grammar.
The sugaring should be self-explanatory.
Let us first see the code, and then I will explain the scenario.

```
### Receive
while not inq.is_full:
    m = take();
    put(inq, m);

### Process
if not is_processing {
    t_t = t_t + 1;
} else {
    t_p = t_p + 1;
}
if t_p == T_p {
    is_processing = false;
    t_p = 0;
    T_p = T_p0;
} else {
    nop;
}
if t_t == T_t {
    is_processing = true;
    t_t = 0
    r = {type: "routing message", payload: "-", destination: BROADCAST};
    if not outq.is_full {
        put(outq, r);
    } else {
        nop;
    }
}
while not inq.is_empty {
    m = get(inq);
    if m.type == "routing message" {
        if is_processing {
            T_p = T_p + 1;
        } else {
            nop;
        }
    } else if m.type == "update" {
        is_processing = true;
        t_t = 0
        r = {type: "routing message", payload: "-", destination: BROADCAST};
        if not outq.is_full {
            put(outq, r);
        } else {
            nop;
        }
    } else {
        nop;
    }
}

### Send
while not outq.is_empty:
    m = get(outq);
    give(m);
```
First, some notes:
1. $\texttt{take}$ and $\texttt{give}$ are keywords to interact with the network. I did not write them in the grammar as I am still not sure how to treat them. Informally, $\texttt{take}$ receives the next request to be delivered by the network, and $\texttt{give}$ gives a request from the output queue to the network.
2. $\texttt{BROADCAST}$ is a keywork to denote that this request must be sent to all of the agents.
3. $r.f$ is sugar to access the field $f$ of request $r$, and $q.$is_full is sugar for checking the size of the queue (similarly for is_empty).

The example here is from a [paper by Floyd and Van Jacobson](https://ee.lbl.gov/papers/sync_94.pdf).
The paper studies synchronization of periodic routing messages.
In a nutshell, this is what happens:
1. Each agent, upon the expiration of its $T_t$ timer enters a listening state, during which it listens for routing messages from other nodes and processes them. Moreover, it sends them a routing message once it enters the listening state.
2. For each message that it receives during the listening state, it spends an extra step in this state, captured by the increment to the timer for this state, $T_p$.
3. Once the $T_p$ timer expires, the agent enters the idle state in which it is waiting for the expiration of the $T_t$ timer, captured by incrementing $t_t$. If it receives messages during this period, it just ignores them.
4. There is a special message of the type "update", that can be injected anytime by the environment. This message abruptly takes the agent to the listening state, no matter the value of the timers.

Metastability in this example happens when a triggering update message is injected into one of the agents, possibly by a network operator playing the role of the environment.
As a result, all of the agents are eventually synchronized, meaning that they send their routing messages at axactly the same time.
This is undesired behavior in a realistic network, as it introduces bursts of packets periodically.
As an analogy to the retry storm example, the self-sustaining loop here is the synchronous firing of routing messages.

This instance of metastability is particularly robust, as it does not depend on concurrency at all: no matter what the ordering of messages is, the system converges to a bad self-sustaining state after the trigger.

# Implementation
I decided to implement this DSL in Rust.
The main reason was that I wanted to learn Rust, and after spending countless hours wrestling with the borrow-checker, I can confirm that one should not toy around with Rust.
I am, by the way, still wrestling as we speak.

The bulk of the interpretation part of the system is ready.
If you remember the [TypeScript Bril interpreter](https://github.com/sampsyo/bril/blob/main/brili.ts), this is essentially similar to that: an infrastructure to define and hold a program, and a long list of "what to do" for each action.
The pipeline first receives a piece of code written in the DSL, parses it according to a parser that I have written using [Lark](https://lark-parser.readthedocs.io/en/stable/) in python, and then uses the interpreter written in Rust to return a result.
The execution engine, on the other hand, is currently missing because the borrow-checker does not like me.

# Challenges
Almost everything in this project has been super challenging, as we have been trying to carve out a DSL from thin air to study metastability.
As such, I am currently in doubt of almost all of the design choices that I have made thus far.
The only comforting point is that I have managed to recreate several cases of metastability with the DSL - analyzed with pen and paper - and that these cases are not contrived: they come from real-life experiences.
Moreover, it seems like the DSL captures most of what is crucial for observing and studying metastability: finite capacity, recurrent cycles of processing, and a concurrent execution setting.

The following list, by no means exhaustive, sheds light on some of the noteworthy challenges:

1. The ownership model of Rust, in tandem with the borrow-checker, is a totally new way of thinking about code for me. Things that are pretty straightforward in other languages become complicated. This problem is exacerbated by the fact that I find it convenient to pack stuff together into a struct, and it is not trivial to me at all how to use a struct's methods on itself with a careful observation of ownership, specially in recursive calls.
2. As I mentioned earlier, I have decided to make the DSL look like NetKAT. My hunch for this was that the general story in our setting looks very similar to the one studied by NetKAT: requests (packets) travelling around, and getting processed by agents (routers) based on their type (port, switch, etc). However, the semantics are currently drastically different. The important abstraction in NetKAT is a packet, and the semantics revolves around a packet: the network is a mapping of a packet to a set of packets. In our DSL, in contrast, the semantics is much more similar to the classic small-step semantics of, say, IMP, where you have a state (or a store), and you apply some operations on it. My hope was to be able to use something similar to the equational inference system baked into NetKAT, via Kleene algebras, to study metastability. I have not yet seen a clear connection.
3. On the long run, one would hope to have a performant interpreter, i.e., one that can interpret large systems in a short span of time. The crux of the difficulty lies in parallelizing the execution in the right way. While there are straightforward ways of attempting this, I have not yet found a good way of doing it.
4. The previous point is exacerbated by the fact that one would also like to have a model-checker that would use the interpreter. Ideally, the model-checker would receive a given system and produce a trace "ending" in a bad self-sustaining loop caused by a trigger. Following the maxim "immature optimization is the root of all evil", I have not spent much time thinking about this yet, but I am expecting some serious issues down the road.