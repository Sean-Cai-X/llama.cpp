(defrule derive-node-has-slice
  (meta-node (node-id ?node-id))
  (slice-instance (bind-node-id ?node-id) (slice-id ?slice-id) (provider-id ?provider-id) (evidence-ref ?evidence-ref))
  =>
  (assert (node-has-slice
    (node-id ?node-id)
    (slice-id ?slice-id)
    (provider-id ?provider-id)
    (evidence-ref ?evidence-ref))))

(defrule derive-graph-governed
  (meta-node (domain-name ?domain-name))
  (meta-inference-rule (rule-id ?rule-id) (is-enabled TRUE))
  =>
  (assert (graph-governed
    (domain-name ?domain-name)
    (rule-id ?rule-id))))

(defrule mark-empty-result-require-human
  (query-context (result-count 0))
  =>
  (assert (admission-decision
    (slice-id "query-context")
    (decision require_human)
    (reason "no-retrieved-slices")
    (rule-id "rule.admission.empty-result")
    (next-action "collect_more_context"))))

(defrule reject-invalid-body-quality
  (slice-instance (slice-id ?slice-id) (body-quality ?quality))
  (test (or (eq ?quality "empty") (eq ?quality "brace_only") (eq ?quality "projection_only_brace") (eq ?quality "polluted")))
  =>
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision reject)
    (reason ?quality)
    (rule-id "rule.admission.invalid-body")
    (next-action "drop_result"))))

(defrule mark-projection-incomplete-require-human
  (slice-instance (slice-id ?slice-id) (projection-incomplete TRUE))
  =>
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision require_human)
    (reason "projection_incomplete")
    (rule-id "rule.admission.projection-incomplete")
    (next-action "audit_projection_only"))))

(defrule mark-truncated-projection-require-human
  (slice-instance (slice-id ?slice-id) (vector-skip-reason ?reason))
  (test (or
    (eq ?reason "append_truncated_projection_incomplete")
    (neq (str-index "projection_only_brace" ?reason) FALSE)))
  =>
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision require_human)
    (reason ?reason)
    (rule-id "rule.admission.truncated-projection")
    (next-action "audit_projection_only"))))

(defrule reject-duplicate-slice
  (slice-instance (slice-id ?slice-id-a) (dedup-hash ?hash&:(neq ?hash "")))
  (slice-instance (slice-id ?slice-id-b) (dedup-hash ?hash))
  (test (neq ?slice-id-a ?slice-id-b))
  =>
  (assert (admission-decision
    (slice-id ?slice-id-b)
    (decision reject)
    (reason "duplicate-dedup-hash")
    (rule-id "rule.admission.duplicate-slice")
    (next-action "link_canonical"))))

(defrule mark-missing-provider-for-repair
  (slice-instance (slice-id ?slice-id) (provider-id ""))
  =>
  (assert (slice-requires-repair
    (slice-id ?slice-id)
    (reason "missing-provider-id")))
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision repair)
    (reason "missing-provider-id")
    (rule-id "rule.admission.missing-provider")
    (next-action "attach_provider_id"))))

(defrule mark-missing-evidence-for-repair
  (slice-instance (slice-id ?slice-id) (evidence-ref ""))
  =>
  (assert (slice-requires-repair
    (slice-id ?slice-id)
    (reason "missing-evidence-ref")))
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision repair)
    (reason "missing-evidence-ref")
    (rule-id "rule.admission.missing-evidence")
    (next-action "attach_evidence_ref"))))

(defrule mark-runtime-evidence-short-term-only
  (slice-instance (slice-id ?slice-id) (provider-id ?provider-id) (evidence-ref ?evidence-ref))
  (test (neq ?provider-id ""))
  (test (eq (str-index "runtime:" ?evidence-ref) 1))
  =>
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision short_term_only)
    (reason "runtime-evidence-only")
    (rule-id "rule.admission.runtime-evidence")
    (next-action "keep_out_of_long_term_memory"))))

(defrule mark-verified-slice-allow
  (slice-instance
    (slice-id ?slice-id)
    (provider-id ?provider-id)
    (evidence-ref ?evidence-ref)
    (body-quality ?quality)
    (projection-incomplete FALSE)
    (vector-skip-reason ?vector-skip-reason)
    (truth-status TRUE))
  (test (neq ?provider-id ""))
  (test (neq ?evidence-ref ""))
  (test (eq ?quality "clean"))
  (test (eq ?vector-skip-reason ""))
  (not (admission-decision (slice-id ?slice-id) (decision reject)))
  (not (admission-decision (slice-id ?slice-id) (decision repair)))
  (not (admission-decision (slice-id ?slice-id) (decision short_term_only)))
  (not (admission-decision (slice-id ?slice-id) (decision require_human)))
  =>
  (assert (admission-decision
    (slice-id ?slice-id)
    (decision allow)
    (reason "verified-slice")
    (rule-id "rule.admission.verified-slice")
    (next-action "admit_to_cognitive_layer"))))

(defrule reject-self-coupling-candidate
  (coupling-candidate
    (candidate-id ?candidate-id)
    (from-slice ?slice-id)
    (to-slice ?slice-id)
    (candidate-type ?candidate-type)
    (score ?score)
    (status CANDIDATE))
  =>
  (assert (coupling-decision
    (candidate-id ?candidate-id)
    (decision REJECT)
    (reason "SELF_COUPLING_NOT_ALLOWED")
    (final-edge-type ?candidate-type)
    (final-confidence ?score)
    (status FINAL))))

(defrule reject-coupling-below-policy-threshold
  (coupling-policy
    (task-type ?task-type)
    (min-confidence ?min-confidence))
  (coupling-candidate
    (candidate-id ?candidate-id)
    (candidate-type ?candidate-type)
    (score ?score&:(< ?score ?min-confidence))
    (status CANDIDATE))
  =>
  (assert (coupling-decision
    (candidate-id ?candidate-id)
    (decision REJECT)
    (reason "COUPLING_SCORE_BELOW_POLICY_THRESHOLD")
    (final-edge-type ?candidate-type)
    (final-confidence ?score)
    (status FINAL))))

(defrule approve-coupling-candidate-by-policy
  (coupling-policy
    (task-type ?task-type)
    (allowed-edge-types $?allowed-before ?candidate-type $?allowed-after)
    (min-confidence ?min-confidence))
  (coupling-candidate
    (candidate-id ?candidate-id)
    (from-slice ?from-slice)
    (to-slice ?to-slice)
    (candidate-type ?candidate-type)
    (score ?score&:(>= ?score ?min-confidence))
    (source ?source)
    (status CANDIDATE))
  (semantic-slice (slice-id ?from-slice) (status VERIFIED))
  (semantic-slice (slice-id ?to-slice) (status VERIFIED))
  (test (neq ?from-slice ?to-slice))
  (not (coupling-decision (candidate-id ?candidate-id) (decision REJECT)))
  =>
  (assert (coupling-decision
    (candidate-id ?candidate-id)
    (decision APPROVE)
    (reason "COUPLING_ALLOWED_BY_POLICY")
    (final-edge-type ?candidate-type)
    (final-confidence ?score)
    (status FINAL)))
  (assert (slice-edge
    (edge-id ?candidate-id)
    (from-slice ?from-slice)
    (to-slice ?to-slice)
    (edge-type ?candidate-type)
    (confidence ?score)
    (source ?source)
    (evidence-ref-a "")
    (evidence-ref-b "")
    (status VERIFIED))))

(defrule reject-fact-claim-without-evidence
  (viewpoint-candidate
    (viewpoint-id ?viewpoint-id)
    (claim-type FACT_CLAIM)
    (status CANDIDATE))
  (not (evidence-binding
    (viewpoint-id ?viewpoint-id)
    (match-status MATCHED)
    (status VERIFIED)))
  =>
  (assert (viewpoint-decision
    (viewpoint-id ?viewpoint-id)
    (decision REJECT)
    (reason "FACT_CLAIM_WITHOUT_VERIFIED_EVIDENCE")
    (final-confidence 0.0)
    (action "block")
    (status FINAL))))

(defrule downgrade-speculation-viewpoint
  (viewpoint-candidate
    (viewpoint-id ?viewpoint-id)
    (claim-type SPECULATION)
    (confidence ?confidence)
    (status CANDIDATE))
  =>
  (assert (viewpoint-decision
    (viewpoint-id ?viewpoint-id)
    (decision DOWNGRADE_TO_HYPOTHESIS)
    (reason "SPECULATION_CANNOT_BE_TREATED_AS_FACT")
    (final-confidence ?confidence)
    (action "label_as_hypothesis")
    (status FINAL))))

(defrule approve-supported-fact-viewpoint
  (viewpoint-candidate
    (viewpoint-id ?viewpoint-id)
    (claim-type FACT_CLAIM)
    (confidence ?confidence)
    (status CANDIDATE))
  (evidence-binding
    (viewpoint-id ?viewpoint-id)
    (match-status MATCHED)
    (status VERIFIED))
  (not (viewpoint-decision (viewpoint-id ?viewpoint-id) (decision REJECT)))
  =>
  (assert (viewpoint-decision
    (viewpoint-id ?viewpoint-id)
    (decision APPROVE)
    (reason "FACT_CLAIM_SUPPORTED_BY_VERIFIED_EVIDENCE")
    (final-confidence ?confidence)
    (action "admit_to_verified_fact_layer")
    (status FINAL))))
