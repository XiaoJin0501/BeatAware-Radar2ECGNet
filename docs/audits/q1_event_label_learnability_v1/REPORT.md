# Q1 event-label learnability Gate 0

Fixed `prominence` delineation on unique MMECG ECG windows. This audits computational label completeness and overlap repeatability; it does not establish clinical validity.

| Fiducial | Availability median | Minimum subject | Repeatability coverage | Repeatability MedAE (ms) | Repeatability P95 (ms) | Complete | Repeatable |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Q | 1.000 | 1.000 | 0.999 | 0.0 | 5.0 | PASS | PASS |
| R | 1.000 | 1.000 | 0.998 | 0.0 | 0.0 | PASS | PASS |
| S | 1.000 | 1.000 | 0.994 | 0.0 | 0.0 | PASS | PASS |
| T | 0.969 | 0.898 | 0.981 | 0.0 | 0.0 | PASS | PASS |

Q<R<S<T violation rate: subject median 0.0000, maximum 0.0000 — PASS.

**Gate 0: PASS**

A pass authorises design of a separate three-fold Event-Head experiment only. It is not manuscript evidence and does not validate Q/S peaks clinically.
