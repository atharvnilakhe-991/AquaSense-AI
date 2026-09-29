# AMDFE Data Quality Assessment Methodology

## Quality Dimensions
- **Completeness ($C$)**: Ratio of non-missing values across feature blocks ($C \in [0, 1]$).
- **Temporal Consistency ($T$)**: Evaluates observation continuity and temporal precedence adherence.
- **Spatial Coverage ($S$)**: Valid spatial coordinate bounds and domain representation.
- **Observation Density ($O$)**: Normalized observation frequency per well.
- **Measurement Quality ($M$)**: Sensor instrumentation precision metadata. Recorded as `unavailable` because no explicit sensor error bars exist in the raw dataset.

## Source vs. Sample-Level Differentiation
Global modality quality measures source-level trustworthiness, while sample-level availability indicators $A_{i,m}$ quantify per-row data completeness for adaptive fusion.
