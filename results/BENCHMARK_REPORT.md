# LIDA-ARC Benchmark Evaluation Report

*Generated: 2025-11-18 06:45:56*

## Summary

| Dataset | Tasks | Tests | Solved | Solve Rate | Avg Accuracy | Avg Time |
|---------|-------|-------|--------|------------|--------------|----------|
| Training | 6 | 6 | 6 | 100.0% | 100.0% | 305ms |
| Evaluation | 4 | 4 | 3 | 75.0% | 81.2% | 305ms |

## Training Dataset Results

| Task ID | Test | Status | Accuracy | Time (ms) |
|---------|------|--------|----------|-----------|
| 00d62c1b | 0 | ✓ Solved | 100.0% | 307 |
| 1e0a9b12 | 0 | ✓ Solved | 100.0% | 304 |
| 2f876c35 | 0 | ✓ Solved | 100.0% | 305 |
| 3c9b0459 | 0 | ✓ Solved | 100.0% | 305 |
| 4be741c5 | 0 | ✓ Solved | 100.0% | 304 |
| 5bd6f4ac | 0 | ✓ Solved | 100.0% | 304 |

## Evaluation Dataset Results

| Task ID | Test | Status | Accuracy | Time (ms) |
|---------|------|--------|----------|-----------|
| 6e82a1ae | 0 | ✓ Solved | 100.0% | 304 |
| 7df24a62 | 0 | ✓ Solved | 100.0% | 306 |
| 8f2ea7aa | 0 | ✓ Solved | 100.0% | 304 |
| 9ecd008a | 0 | ✗ Failed | 25.0% | 305 |

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
- Average solve time (training): 305ms
- Average solve time (evaluation): 305ms

### Pattern Types Solved

Based on successful tasks:
- ✓ 90-degree rotation
- ✓ Horizontal reflection
- ✓ Vertical reflection
- ✓ Color remapping
