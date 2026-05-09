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
