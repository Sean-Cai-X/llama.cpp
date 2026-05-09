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
