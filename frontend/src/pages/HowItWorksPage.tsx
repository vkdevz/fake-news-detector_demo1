import React from 'react';
import { 
  FileText, Sparkles, Cpu, Search, Globe, Award, 
  GitCompare, History, ShieldCheck, CheckCircle2 
} from 'lucide-react';

export const HowItWorksPage: React.FC = () => {
  const steps = [
    {
      num: 1,
      title: 'Content Ingestion & Safe Normalization',
      icon: FileText,
      desc: 'Ingests raw text, URLs (with strict SSRF defense blocking loopback/metadata IPs), or single claims. Cleans boilerplate, extracts language (English, Hindi, Hinglish), and normalizes whitespace.'
    },
    {
      num: 2,
      title: 'Claim Extraction & Decomposition',
      icon: Sparkles,
      desc: 'Decomposes complex compound sentences into atomic factual assertions. Filters out subjective opinions, rhetorical remarks, and satirical parody.'
    },
    {
      num: 3,
      title: 'ML Classification (Linguistic Prior)',
      icon: Cpu,
      desc: 'Applies TF-IDF n-gram vectorization and Linear SVM / Naive Bayes classifier to obtain a prior probability based on vocabulary and sensational stylistic markers.'
    },
    {
      num: 4,
      title: 'Fact-Check Search & Matching',
      icon: Search,
      desc: 'Queries Google Fact Check Claim Search API and local accredited fact-check archives. Evaluates semantic similarity to ensure high-relevance matches.'
    },
    {
      num: 5,
      title: 'Multi-Query Web Evidence Retrieval',
      icon: Globe,
      desc: 'Generates diverse search queries (exact claim, entity-event, official domain query, contradiction query). Retrieves documentation from primary agencies and trusted reporting.'
    },
    {
      num: 6,
      title: 'Source Evaluation & Deduplication',
      icon: Award,
      desc: 'Rates sources by domain authority (.gov, .edu, accredited news: 0.90-0.98), publication freshness, and directness. Detects syndicated wire copies so multiple reprints do not bias source count.'
    },
    {
      num: 7,
      title: 'NLI Contradiction & Stance Detection',
      icon: GitCompare,
      desc: 'Analyzes natural language inference relationships (SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, OUTDATED, NEUTRAL) between claim assertions and retrieved document excerpts.'
    },
    {
      num: 8,
      title: 'Temporal Context & Timeline Reasoning',
      icon: History,
      desc: 'Aligns claim dates against event and publication dates. Prevents historically true facts that have been superseded from being wrongly labeled as malicious fake news.'
    },
    {
      num: 9,
      title: 'Explainable Hybrid Verdict Engine',
      icon: ShieldCheck,
      desc: 'Fuses ML prior probability, weighted evidence support scores, contradiction weights, and temporal validity into an 11-category evidence-grounded determination with calibrated confidence.'
    },
    {
      num: 10,
      title: 'Traceable Evidence-Grounded Citations',
      icon: CheckCircle2,
      desc: 'Produces transparent explanations directly referencing retrieved excerpts and URLs, defusing prompt-injection payloads and guaranteeing auditability.'
    },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          How TruthLens Works
        </h2>
        <p className="text-sm sm:text-base text-slate-400">
          The 10-stage hybrid architecture combining classical ML with multi-source factual verification.
        </p>
      </div>

      <div className="space-y-4">
        {steps.map((step) => {
          const Icon = step.icon;
          return (
            <div
              key={step.num}
              className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm flex items-start gap-5 hover:border-slate-700 transition-colors"
            >
              <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-400 flex items-center justify-center shrink-0">
                <Icon className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono font-bold text-teal-400">STAGE {step.num}</span>
                  <h3 className="text-base font-bold text-white">{step.title}</h3>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed font-sans">{step.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
