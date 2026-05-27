(deftemplate meta-node
  (slot node-id)
  (slot parent-id)
  (slot anchor-level (type INTEGER))
  (slot is-immutable)
  (slot node-type)
  (slot lifecycle)
  (slot domain-name))

(deftemplate concept-node
  (slot node-id)
  (slot parent-id)
  (slot anchor-level (type INTEGER))
  (slot concept-name)
  (slot core-definition))

(deftemplate slice-instance
  (slot link-id)
  (slot slice-id)
  (slot bind-node-id)
  (slot info-weight)
  (slot truth-status)
  (slot is-reversible)
  (slot similarity-to-core)
  (slot slice-text)
  (slot embedding-hash)
  (slot dedup-hash)
  (slot source-type)
  (slot provider-id)
  (slot evidence-ref)
  (slot body-quality)
  (slot projection-incomplete)
  (slot vector-skip-reason)
  (slot browser-visible-summary))

(deftemplate meta-inference-rule
  (slot rule-id)
  (slot inference-type)
  (slot source-node-type)
  (slot target-node-type)
  (slot max-diffuse-depth (type INTEGER))
  (slot max-slice-per-node (type INTEGER))
  (slot min-similarity-threshold (type FLOAT))
  (slot is-enabled))

(deftemplate query-context
  (slot query)
  (slot retrieval-mode)
  (slot result-count (type INTEGER)))

(deftemplate node-has-slice
  (slot node-id)
  (slot slice-id)
  (slot provider-id)
  (slot evidence-ref))

(deftemplate graph-governed
  (slot domain-name)
  (slot rule-id))

(deftemplate slice-requires-repair
  (slot slice-id)
  (slot reason))

(deftemplate admission-decision
  (slot slice-id)
  (slot decision)
  (slot reason)
  (slot rule-id)
  (slot next-action))

(deftemplate semantic-slice
  (slot slice-id)
  (slot file-id)
  (slot source-uri)
  (slot start-line (type INTEGER))
  (slot end-line (type INTEGER))
  (slot text-hash)
  (slot language)
  (slot symbol-scope)
  (slot status))

(deftemplate slice-edge
  (slot edge-id)
  (slot from-slice)
  (slot to-slice)
  (slot edge-type)
  (slot confidence (type FLOAT))
  (slot source)
  (slot evidence-ref-a)
  (slot evidence-ref-b)
  (slot status))

(deftemplate coupling-candidate
  (slot candidate-id)
  (slot from-slice)
  (slot to-slice)
  (slot candidate-type)
  (slot score (type FLOAT))
  (slot source)
  (slot status))

(deftemplate coupling-policy
  (slot task-type)
  (multislot allowed-edge-types)
  (slot max-depth (type INTEGER))
  (slot max-context-slices (type INTEGER))
  (slot min-confidence (type FLOAT))
  (slot token-budget (type INTEGER)))

(deftemplate coupling-decision
  (slot candidate-id)
  (slot decision)
  (slot reason)
  (slot final-edge-type)
  (slot final-confidence (type FLOAT))
  (slot status))

(deftemplate llama-output-candidate
  (slot request-id)
  (slot trace-id)
  (slot model-task-id)
  (slot source-slice-id)
  (slot output-hash)
  (slot status))

(deftemplate viewpoint-candidate
  (slot viewpoint-id)
  (slot request-id)
  (slot trace-id)
  (slot model-task-id)
  (slot source-slice-id)
  (slot claim-text)
  (slot claim-type)
  (slot subject)
  (slot predicate)
  (slot object)
  (slot confidence (type FLOAT))
  (slot status))

(deftemplate evidence-binding
  (slot viewpoint-id)
  (slot source-type)
  (slot source-id)
  (slot evidence-ref)
  (slot evidence-hash)
  (slot match-status)
  (slot status))

(deftemplate viewpoint-validation
  (slot viewpoint-id)
  (slot schema-valid)
  (slot evidence-valid)
  (slot rule-valid)
  (slot conflict-status)
  (slot confidence-valid)
  (slot scope-valid)
  (slot validation-status)
  (slot reject-reason))

(deftemplate viewpoint-decision
  (slot viewpoint-id)
  (slot decision)
  (slot reason)
  (slot final-confidence (type FLOAT))
  (slot action)
  (slot status))
