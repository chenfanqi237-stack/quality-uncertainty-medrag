# Stance uncertainty evaluation

**FINAL EVALUATION: completeness gate passed**

Mode: `FINAL_EVALUATION`
Checkpoint SHA256: `356cdb4f30de17952e22176bc6127b41384bae3a795baf8c3724612162ee4f34`
Reference provenance: AI-assisted adjudicated stance reference set; not independent clinician annotation.

The ten-seed values are empirical self-consistency frequencies, not calibrated probabilities. Higher uncertainty is scored as more likely to be a frozen hard-stance error.

## Completeness

- Valid samples: 600 / 600
- Failed samples: 0
- Missing samples: 0
- Complete pairs: 60
- Incomplete pairs: 0
- Evaluable completed pairs: 60

## Error detection

| Score | AUROC | Average precision | Error prevalence |
|---|---:|---:|---:|
| u_3 | 0.5814 | 0.3823 | 0.2833 |
| u_directional | 0.3222 | 0.2447 | 0.2833 |

## Reference class distribution

| Class | Count |
|---|---:|
| SUPPORT | 8 |
| CONTRADICT | 7 |
| IRRELEVANT | 45 |

## Aggregation experiment preparation

The frozen 60-pair input projection has neither evidence_type metadata (needed by the existing frozen quality scorer) nor a frozen quality_score. It is a stance-reference subset (50 question-option groups), not a declared complete retrieval-evidence set for every answer option. Quality-weighted aggregation cannot be compared without the frozen evidence grouping and quality metadata.

No new inference, threshold optimization, aggregation comparison, or end-to-end MedQA answer evaluation was performed.
