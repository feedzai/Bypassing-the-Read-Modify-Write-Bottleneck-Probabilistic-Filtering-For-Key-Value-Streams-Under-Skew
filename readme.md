# Decoupling Inference from State Updates in Low-Latency Feature Engines via Probabilistic Thinning

Welcome to the reporducibility instructions for Decoupling Inference from State
Updates in Low-Latency Feature Engines via Probabilistic Thinning.

*Abstract*

Streaming data systems increasingly underpin Machine Learning workflows that
maintain large numbers of continuously updated aggregations. In production
settings, each incoming event typically triggers read-modify-write operations
to persistent storage, making high-frequency state updates a dominant source of
latency, contention, and operational cost. In this work, we show how to decouple
inference from state persistence in streaming Machine Learning pipelines via
probabilistic thinning: every event is scored, but persistent state updates are
only triggered by *the most informative events*. We demonstrate that such
thinning can be implemented without local in-memory control state or
coordination, relying exclusively on approximate statistics retrieved from
persistent key-value stores. We model the resulting stochastic processes, derive
bounds on filtering rates, and show that common time-based aggregations remain
unbiased under variance-aware formulations. Thus, they do not accumulate
systemic errors. We implement this approach in a real-world transaction
monitoring system, and demonstrate substantial reductions in storage
Input/Output and serialization overhead, often improving downstream fraud
detection accuracy; in our example, we exclude over 90% of events from the
persistence path while consistently outperforming the baseline.

# Reproducing the results

1. Follow the instructions for the [server](systems_experimentation/server/README.md)
2. Follow the instructions for the [injector](systems_experimentation/injector/README.md)
