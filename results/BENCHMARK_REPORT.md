# LIDA-ARC Benchmark Evaluation Report

*Generated: 2025-11-18 07:19:44*

## Summary

| Dataset | Tasks | Tests | Solved | Solve Rate | Avg Accuracy | Avg Time |
|---------|-------|-------|--------|------------|--------------|----------|
| Training | 6 | 6 | 6 | 100.0% | 100.0% | 304ms |
| Evaluation | 4 | 4 | 3 | 75.0% | 81.2% | 303ms |

## Training Dataset Results

| Task ID | Test | Status | Accuracy | Time (ms) |
|---------|------|--------|----------|-----------|
| 00d62c1b | 0 | ✓ Solved | 100.0% | 304 |
| 1e0a9b12 | 0 | ✓ Solved | 100.0% | 303 |
| 2f876c35 | 0 | ✓ Solved | 100.0% | 304 |
| 3c9b0459 | 0 | ✓ Solved | 100.0% | 305 |
| 4be741c5 | 0 | ✓ Solved | 100.0% | 305 |
| 5bd6f4ac | 0 | ✓ Solved | 100.0% | 304 |

## Evaluation Dataset Results

| Task ID | Test | Status | Accuracy | Time (ms) |
|---------|------|--------|----------|-----------|
| 6e82a1ae | 0 | ✓ Solved | 100.0% | 303 |
| 7df24a62 | 0 | ✓ Solved | 100.0% | 303 |
| 8f2ea7aa | 0 | ✓ Solved | 100.0% | 303 |
| 9ecd008a | 0 | ✗ Failed | 25.0% | 303 |

## Analysis

### Key Findings

**Training Set Performance:**
- Solved 6/6 tasks (100.0% solve rate)
- Average accuracy: 100.0%

**Evaluation Set Performance:**
- Solved 3/4 tasks (75.0% solve rate)
- Average accuracy: 81.2%

**Generalization:**
- Relative performance on eval vs training: 75.0%

**Efficiency:**
- Average solve time (training): 304ms
- Average solve time (evaluation): 303ms

### Pattern Types Solved

Based on successful tasks:
- ✓ 90-degree rotation
- ✓ Horizontal reflection
- ✓ Vertical reflection
- ✓ Color remapping
