# Executive Briefing: FD-LLM System

**Project:** Hybrid Fault Detection & Explanation for Slurry Pipelines  
**Date:** October 13, 2025  
**Status:** ✅ Complete Working Prototype

---

## 1. Executive Summary

The FD-LLM project successfully delivers a **working prototype of a next-generation monitoring system** for our slurry pipelines. This system addresses the critical need for both immediate fault detection and expert-level analysis of operational issues.

Our solution is a **hybrid AI model** that combines two powerful technologies:
1.  A **real-time, high-speed classifier** that instantly detects anomalies in sensor data.
2.  An **AI-powered language model** that provides clear, human-like explanations for *why* a fault occurred, complete with evidence and recommended actions.

This prototype proves the viability of an automated system that not only tells us *when* something is wrong but also provides the expert-level insights needed to fix it quickly. The immediate business benefits include **reduced downtime, improved operational efficiency, and enhanced safety**.

---

## 2. The Challenge: Beyond Simple Alarms

Traditional monitoring systems often generate a high volume of alarms without providing context. This leaves operators to manually diagnose complex issues, leading to:
*   **Delayed Responses:** Time spent interpreting sensor data is time the pipeline is operating inefficiently or unsafely.
*   **Inconsistent Diagnoses:** The quality of analysis depends heavily on the experience of the on-duty operator.
*   **Lost Knowledge:** Expert diagnostic knowledge is difficult to scale and retain.

We needed a system that could automate not just detection, but also the expert reasoning process.

---

## 3. Our Solution: The FD-LLM Hybrid System

FD-LLM is a two-part system designed to mimic the workflow of an expert process engineer.

#### **Part 1: The Watchdog (MultiROCKET Classifier)**
This is a state-of-the-art time-series model that continuously monitors pipeline sensor data. It's trained to recognize complex patterns indicating specific faults (like dilution, blockages, or settling).

*   **Function:** Real-time fault detection.
*   **Speed:** Near-instantaneous (<1 millisecond per analysis).
*   **Analogy:** Think of it as an EKG for the pipeline, constantly watching the heartbeat and flagging any irregular rhythms the moment they occur.

#### **Part 2: The Expert Analyst (Mistral-7B LLM)**
When the Watchdog detects a fault, it triggers the Expert Analyst. This Large Language Model (LLM) receives the relevant sensor data and provides a comprehensive diagnosis in plain English.

*   **Function:** In-depth root cause analysis and explanation.
*   **Output:** A structured report including diagnosis, supporting evidence, physical cross-checks, and recommended actions.
*   **Analogy:** This is our 24/7 on-call process engineer who can instantly analyze the EKG results and explain the underlying condition.

**Built-in Reliability:** To ensure the LLM's explanations are trustworthy, we use a "self-consistency" method. The system generates five independent diagnoses and votes on the most logical conclusion, effectively getting a second, third, and fourth opinion automatically.

---

## 4. Key Features & Capabilities

*   **End-to-End Automation:** From raw sensor data to actionable insights with no manual intervention.
*   **Real-Time Performance:** The classifier is fast enough for live production monitoring.
*   **Rich, Contextual Explanations:** Moves beyond simple alerts to provide genuine understanding of operational issues.
*   **Data-Driven Evidence:** All claims made by the AI are backed by specific, numeric evidence from the sensor data.
*   **Physical Validation:** The system performs physics-based checks (e.g., mass balance) to ensure its conclusions are grounded in reality.
*   **Scalable Architecture:** Designed with a flexible architecture where the fast classifier can run locally (on-site) and the computationally-heavy LLM can run on cost-effective cloud GPUs.

---

## 5. Current Status & Demo Performance

The system is a **complete, functional prototype**. We have successfully trained it on a dataset from our operations and generated both predictions and explanations.

*   **Classifier Accuracy:** The prototype achieves **75% accuracy** on the available test data. This is a strong result for a first iteration and proves the model can learn meaningful patterns.
*   **LLM Explanation Quality:** The LLM provides high-quality, relevant explanations with a **100% agreement rate** in our test set, demonstrating high reliability.

**Live Demo:** A fully interactive command-line demo (`demo.py`) has been created to showcase the entire system's capabilities using the pre-trained models.

---

## 6. Business Value & Potential ROI

*   **Reduce Downtime:** By providing immediate root cause analysis, operators can resolve issues faster, minimizing non-productive time.
*   **Improve Efficiency:** The system can detect subtle inefficiencies (like gradual dilution) that might otherwise go unnoticed.
*   **Enhance Safety:** Proactive detection of hazardous conditions (e.g., potential blockages) allows for preventative action.
*   **Democratize Expertise:** Captures and scales the knowledge of senior process engineers, making it available 24/7 across all sites.
*   **Data-Driven Decision Making:** Provides a consistent, objective source of truth for operational decisions.

---

## 7. Path to Production

This prototype has successfully de-risked the core technical challenges. To move this system into production, a focused effort on data quality, model hardening, and advanced validation is required. The following outlines the clear, actionable steps to achieve this.

### Phase 1: Classifier Enhancement (Target: ≥90% Accuracy)

The highest priority is to improve the classifier's accuracy from its current 75% to a production-ready target of over 90%. This will be achieved through a data-centric approach:

1.  **Source High-Quality Training Data:**
    *   **Action:** Acquire CSV data from standard operational periods, avoiding known anomalies or atypical process conditions used in the prototype.
    *   **Goal:** Create a dataset that reflects real-world conditions, with an expected distribution of **85-95% Normal** operations and **5-15% genuine faults**.

2.  **Verify and Refine Data Labels:**
    *   **Action:** Conduct a review of the existing process for assigning fault labels to data.
    *   **Opportunity:** Leverage the LLM component to assist in semi-automating the labeling of new data, ensuring consistency and capturing expert knowledge efficiently.

3.  **Implement a Balanced Training Strategy:**
    *   **Action:** Address the natural class imbalance in the data. Use stratified sampling to ensure training and test sets are representative.
    *   **Goal:** Prevent the model from being biased towards the "Normal" state and improve its ability to detect rare but critical faults.

4.  **Retrain and Validate:**
    *   **Action:** Retrain the MultiROCKET classifier on the new, curated dataset.
    *   **Success Metrics:** Achieve **≥90% overall accuracy** and, critically, high **per-class recall** to ensure we don't miss infrequent fault types. Validate performance across different time periods to confirm generalization.

### Phase 2: Production Hardening & Validation

Once the classifier meets the accuracy target, we must ensure the entire system is reliable and robust enough for a live environment.

1.  **Comprehensive Automated Testing:**
    *   **Action:** Implement a full suite of unit and integration tests (`test_loaders`, `test_features`, `test_integration`).
    *   **Goal:** Ensure code quality, prevent regressions, and automate validation of the entire data-to-explanation pipeline.

2.  **Systematic Robustness Testing:**
    *   **Action:** Execute a series of stress tests that simulate real-world hardware and data issues.
    *   **Scenarios:**
        *   **Sensor Dropout:** Test system performance when key sensors (pressure, temperature) temporarily fail.
        *   **Noise Injection:** Artificially add noise to the data to test model stability.
        *   **Calibration Drift:** Simulate gradual sensor degradation to measure impact on accuracy.

### Phase 3: Advanced Capabilities & Generalization

With a robust and accurate model, the final phase focuses on scaling its intelligence and applicability.

1.  **Cross-Machine Generalization:**
    *   **Action:** Test the trained model on data from different slurry pipelines or operational sites without retraining.
    *   **Goal:** Confirm that the model has learned fundamental fault principles rather than the specifics of a single piece of equipment.

2.  **Fine-Tune the LLM Explainer:**
    *   **Action:** Move from the current pre-trained LLM to a model fine-tuned on our own specific operational data and expert reports.
    *   **Benefit:** This will result in even more precise, domain-aware explanations that use our internal terminology and reflect our specific operational procedures.

This structured approach ensures we build upon the successful prototype to deliver a reliable, accurate, and scalable production system.
