export type VerdictType = 
  | 'SUPPORTED'
  | 'LIKELY_TRUE'
  | 'PARTIALLY_TRUE'
  | 'MISLEADING'
  | 'UNSUPPORTED'
  | 'LIKELY_FALSE'
  | 'FALSE'
  | 'OUTDATED'
  | 'UNVERIFIABLE'
  | 'SATIRE'
  | 'OPINION';

export type RelationshipType = 
  | 'SUPPORTS'
  | 'CONTRADICTS'
  | 'PARTIALLY_SUPPORTS'
  | 'NEUTRAL'
  | 'OUTDATED'
  | 'UNRELATED';

export interface PipelineStep {
  step_key: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped';
  details?: string;
  duration_ms?: number;
}

export interface EvidenceItem {
  id: string;
  claim_id: string;
  title: string;
  url: string;
  publisher?: string;
  domain?: string;
  source_type: string;
  authority_score: number;
  freshness_score: number;
  publication_date?: string;
  relationship: RelationshipType;
  relevance_score: number;
  stance_confidence: number;
  excerpt: string;
  reasoning?: string;
  temporal_status?: string;
  cluster_id?: string;
  is_syndicated?: boolean;
  syndicate_count?: number;
  syndicated_domains?: string[];
  independence_score?: number;
}

export interface FactCheckItem {
  matched_claim: string;
  rating: string;
  publisher: string;
  review_date?: string;
  url?: string;
  summary?: string;
  semantic_similarity: number;
}

export interface ClaimVerdict {
  claim_id: string;
  claim_text: string;
  verdict: VerdictType;
  confidence: number;
  confidence_label: string;
  support_score: number;
  contradiction_score: number;
  reason: string;
  supporting_evidence_count: number;
  contradicting_evidence_count: number;
  evidences: EvidenceItem[];
}

export interface OverallVerdict {
  verdict: VerdictType;
  confidence: number;
  confidence_label: string;
  ml_probability?: number;
  ml_verdict?: string;
  support_score: number;
  contradiction_score: number;
  explanation: string;
  has_conflicting_evidence: boolean;
  is_outdated: boolean;
  is_misleading: boolean;
}

export interface TimelineEvent {
  event_type: string;
  title: string;
  date: string;
  description: string;
}

export interface VerificationResponse {
  request_id: string;
  input_type: string;
  raw_input: string;
  status: string;
  created_at: string;
  language: string;
  pipeline_steps: PipelineStep[];
  overall_verdict: OverallVerdict;
  claims: ClaimVerdict[];
  all_evidence: EvidenceItem[];
  fact_checks: FactCheckItem[];
  timeline_events: TimelineEvent[];
  source_distribution: Record<string, number>;
  is_demo_mode: boolean;
}

export interface DemoScenario {
  id: string;
  title: string;
  category: string;
  expected_verdict: VerdictType;
  input_type: 'text' | 'url' | 'claim';
  text: string;
  description: string;
}
