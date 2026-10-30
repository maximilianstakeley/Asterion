# Asterion

### Adaptive System Monitoring, Network Security Analysis, and AI-Assisted Decision Support

Asterion is an actively developed defensive monitoring and intelligent decision-support framework designed to observe computer hardware, operating systems, software processes, and network activity.

The project integrates system telemetry collection, anomaly detection, behavioral analysis, statistical modeling, reinforcement learning, and safety-constrained decision-making into a modular architecture.

Its objective is to develop an adaptive system capable of identifying abnormal operating conditions, evaluating supporting evidence, generating potential corrective actions, and assessing those actions through deterministic safety controls.

**Development status:** Active development / Experimental

**Execution policy:** Live system modifications are disabled.

---

## 1. Project Overview

Asterion is designed around a centralized analysis and decision architecture supported by independently operating monitoring agents.

The system separates observation, detection, analysis, decision-making, safety evaluation, and execution to maintain traceability and prevent analytical outputs from automatically becoming operational commands.

The architecture currently incorporates:

- Host resource monitoring and telemetry collection.
- Network traffic and connection monitoring.
- Statistical and rule-based anomaly detection.
- Behavioral and text-based classification.
- Evidence-supported root-cause hypotheses.
- Deterministic decision policies.
- Experimental reinforcement-learning policies.
- Safety validation and authorization controls.
- Persistent telemetry and decision auditing.

## 2. System Architecture

The intended processing pipeline is:

```text
          MONITORED SYSTEMS
                  |
                  v
         TELEMETRY COLLECTORS
                  |
                  v
          CENTRAL FASTAPI API
                  |
                  v
       OBSERVATION VALIDATION
                  |
                  v
          SQLITE PERSISTENCE
                  |
                  v
       DETECTION AND ANALYSIS
                  |
          +-------+-------+
          |               |
          v               v
    ANOMALY DETECTION   BEHAVIORAL
                       CLASSIFICATION
          |               |
          +-------+-------+
                  |
                  v
        ROOT-CAUSE ANALYSIS
                  |
                  v
        EVIDENCE GENERATION
                  |
                  v
       CANDIDATE GENERATION
                  |
                  v
       POLICY SELECTION
          /           \
         v             v
   BASELINE POLICY   PPO POLICY
          \           /
           \         /
                  |
                  v
           SAFETY GATE
                  |
                  v
          EXECUTOR INTERFACE
                  |
                  v
          DRY-RUN / BLOCKED
                  |
                  v
          DECISION AUDIT
```

The reinforcement-learning policy is optional. The deterministic baseline provides a policy-selection alternative when reinforcement learning is unavailable or not configured.

## 3. Implemented Components

### 3.1 System Telemetry

Asterion incorporates independent telemetry collection using Python and psutil.

Currently supported measurements include:

- CPU utilization.
- Physical memory utilization.
- Disk utilization and filesystem identity.
- System load averages.
- Network bytes transmitted and received.
- Network packets transmitted and received.
- Aggregate network throughput.
- Per-interface network counters and throughput.
- Network connection counts and connection states when operating-system permissions permit.

Telemetry observations include timestamps, host identifiers, agent identifiers, measurement units, and collection statuses.

Unavailable measurements are explicitly represented rather than substituted with fabricated values.

### 3.2 Network Monitoring and Security

The network monitoring architecture supports:

- Aggregate network traffic measurements.
- Per-interface traffic monitoring.
- Counter-based throughput calculations.
- Network event and flow ingestion.
- Structured network security records.
- Record provenance and integrity fingerprints.
- Evidence-based network optimization recommendations.

Network security records are treated as observations rather than automatic determinations of malicious activity.

### 3.3 Anomaly Detection

Asterion incorporates several detection approaches:

**Rule-based detection**
- Resource utilization thresholds.
- Abnormal CPU, memory, and disk utilization.

**Time-window detection**
- Persistent resource anomalies.
- Repeated threshold violations.
- Rapid metric changes.
- Insufficient observation coverage.

**Multivariate detection**
- Joint analysis of multiple resource measurements.
- Median-based normalization.
- Median absolute deviation (MAD).
- Robust statistical anomaly scoring.

Detection outputs provide supporting evidence and investigative indicators. Statistical anomalies are not equivalent to verified security incidents.

### 3.4 Behavioral Analysis

The behavioral analysis components incorporate text classification and structured observation analysis.

The architecture supports:

- Classification of supplied text.
- Detection of potentially suspicious descriptions.
- Evidence extraction.
- Classification confidence reporting.
- Analyst feedback collection.

Machine-learning classification is dependent on an available, compatible model artifact. Rule-based detection remains available independently of that artifact.

### 3.5 Root-Cause Analysis

The diagnostic architecture evaluates detected findings and generates potential explanations.

Hypotheses are associated with supporting evidence and uncertainty information.

The objective is to distinguish observed symptoms from possible underlying causes rather than automatically assigning an unsupported explanation to an event.

### 3.6 Decision Engine

The decision engine translates analytical evidence into candidate actions.

Currently implemented capabilities include:

- Evidence-derived candidate generation.
- Candidate validation.
- Deterministic action prioritization.
- Risk-score validation.
- Policy selection.
- Decision serialization.
- Safety evaluation.
- Decision auditing.

The deterministic policy provides a baseline against which experimental reinforcement-learning policies can be evaluated.

### 3.7 Reinforcement Learning

Asterion incorporates an experimental reinforcement-learning subsystem using:

- Gymnasium.
- Stable-Baselines3.
- Proximal Policy Optimization (PPO).
- Custom decision environments.
- Normalized system-state representations.
- Measurement-availability indicators.
- Policy evaluation and comparison.

The reinforcement-learning environment currently uses synthetic simulation rather than direct interaction with production computer systems.

The PPO policy is constrained to the candidate actions generated by the evidence-based decision pipeline.

Its outputs are proposals, not execution authorization.

## 4. Safety and Execution Architecture

Safety is an independent component of the decision pipeline.

The current architecture incorporates:

- Candidate-action validation.
- Authorization checks.
- Resource-budget validation interfaces.
- Action-precondition validation interfaces.
- Risk constraints.
- Fail-closed behavior.
- Explicit execution-state reporting.
- Decision audit records.

**Live execution is disabled by design.**

The executor does not implement live process control, network reconfiguration, file deletion, or workload modification.

A permitted safety verdict does not independently authorize execution.

Simulated execution requires an explicit simulation adapter and remains separate from real system operations.

## 5. Technology Stack

| Technology | Application |
|---|---|
| Python | Core development language |
| FastAPI | Central API |
| Pydantic | Data validation and contracts |
| psutil | Hardware and operating-system telemetry |
| SQLite | Persistent storage |
| NumPy | Numerical computation |
| Gymnasium | Reinforcement-learning environments |
| Stable-Baselines3 | PPO implementation |
| Joblib | Machine-learning artifact loading |
| Pytest | Automated testing |

## 6. API Overview

The FastAPI application provides endpoints for telemetry ingestion, analysis, detection, and decision retrieval.

| Endpoint | Function |
|---|---|
| `GET /health` | Public application health check |
| `GET /v1/status` | System and database counts |
| `POST /v1/telemetry` | Submit host telemetry |
| `GET /v1/telemetry/timeseries` | Retrieve historical telemetry |
| `POST /v1/analyze` | Analyze supplied observations |
| `GET /v1/findings` | Retrieve detected findings |
| `GET /v1/decisions` | Retrieve decision audit records |
| `POST /v1/feedback` | Submit analyst feedback |
| `POST /v1/detection/time-window` | Run time-window detection |
| `POST /v1/detection/multivariate` | Run multivariate detection |
| `POST /v1/network-security/events` | Submit network security records |
| `GET /v1/network-security/events` | Retrieve network security records |

All endpoints other than `/health` require Bearer-token authentication.

## 7. Data Integrity and Measurement Principles

Asterion follows several important data-handling principles:

1. **Measurement integrity:** Values must originate from actual observations or explicitly identified simulations.
2. **Missing-data preservation:** Unavailable measurements must not be interpreted as measured zero values.
3. **Timestamped observations:** Measurements retain temporal context.
4. **Evidence traceability:** Findings and decisions should retain their supporting observations.
5. **Explicit uncertainty:** Analytical hypotheses must not be represented as established facts.
6. **Execution separation:** Policy selection and safety evaluation remain separate from execution.
7. **Auditable decisions:** Decision outcomes and failures are recorded for subsequent examination.

## 8. Development Roadmap

The following are development objectives, not claims that the capabilities are already complete.

- [x] Initial host telemetry collection.
- [x] Structured telemetry validation.
- [x] Centralized API ingestion.
- [x] Persistent telemetry storage.
- [x] Historical telemetry retrieval.
- [x] Rule-based anomaly detection.
- [x] Time-window anomaly detection.
- [x] Multivariate anomaly detection.
- [x] Deterministic baseline policy.
- [x] Reinforcement-learning simulation environment.
- [x] Safety-constrained, non-executing action interface.
- [x] Structured decision auditing.
- [ ] Unified multi-host operational deployment.
- [ ] Expanded network security dataset integration.
- [ ] Long-term behavioral baseline modeling.
- [ ] Expanded RL benchmarking against deterministic policies.
- [ ] Improved model evaluation and promotion workflows.
- [ ] Production-grade identity and access management.
- [ ] Comprehensive deployment and configuration documentation.
- [ ] End-to-end operational validation across multiple hosts.

## 9. Current Limitations

Asterion remains an experimental development project.

Important limitations include:

- Reinforcement learning is evaluated using synthetic environments.
- Connection-level monitoring depends on operating-system permissions.
- Statistical anomaly scores are not calibrated intrusion probabilities.
- Machine-learning classification requires compatible model artifacts.
- Safety checks intentionally prevent progression when required validation components are unavailable.
- Production deployment, distributed coordination, and live remediation have not been established.

No claim of autonomous threat prevention or production-grade cybersecurity protection is made.

## 10. Installation and Development

The repository currently requires its Python package structure and dependency manifest to be standardized before a reproducible installation procedure can be guaranteed.

The intended development environment uses Python 3.12 or a compatible Python version.

Core development dependencies include:

- FastAPI
- Uvicorn
- Pydantic
- psutil
- Requests
- python-dotenv
- Joblib
- NumPy
- Pytest

Optional reinforcement-learning dependencies include Gymnasium and Stable-Baselines3.

See the repository's development configuration and dependency manifest when available.

## 11. Project Objective

The long-term objective of Asterion is to develop a modular, adaptive monitoring and decision-support framework capable of continuously evaluating system behavior, identifying deviations from expected operating conditions, correlating evidence across multiple sources, and producing explainable recommendations.

The broader architectural direction is to integrate statistical analysis, machine learning, reinforcement learning, and deterministic safety controls into a unified system.

The emphasis is on measurable observations, reproducible analysis, explainable decisions, and controlled system interaction.

---

**Project:** Asterion  
**Repository:** https://github.com/maximilianstakeley/Asterion  
**Development:** Active / Experimental  
**Execution:** Disabled by design
