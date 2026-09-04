import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { VerificationResponse } from '../types/verification';

interface VerificationTraceProps {
  result: VerificationResponse;
}

export const VerificationTrace: React.FC<VerificationTraceProps> = ({ result }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const mlFakeProb = result.overall_verdict.ml_probability ?? 0.50;
  const supportScore = result.overall_verdict.support_score;
  const contraScore = result.overall_verdict.contradiction_score;

  return (
    <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 sm:p-6 transition-colors">
      <div 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            Verification trace
          </h3>
          <p className="text-xs text-slate-500 dark:text-neutral-400 mt-0.5">
            Stage-by-stage audit from claim ingestion to final determination.
          </p>
        </div>

        <button 
          className="flex items-center gap-1 text-xs text-slate-500 dark:text-neutral-400 hover:text-slate-800 dark:hover:text-white font-medium px-2 py-1 rounded"
          aria-expanded={isOpen}
        >
          <span>{isOpen ? 'Hide trace' : 'Show trace'}</span>
          {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {isOpen && (
        <div className="mt-5 pt-5 border-t border-slate-100 dark:border-neutral-800 space-y-4 text-xs font-sans animate-fadeIn">
          {/* Stage 1: Input Ingestion */}
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-neutral-400 mb-1.5 font-medium">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 flex items-center justify-center text-[10px] font-mono">1</span>
              <span>Input ({result.input_type.toUpperCase()})</span>
            </div>
            <p className="text-slate-800 dark:text-zinc-200 bg-white dark:bg-[#050505] p-2.5 rounded border border-slate-200/60 dark:border-neutral-800 font-mono text-[11px] leading-relaxed">
              "{result.raw_input}"
            </p>
          </div>

          {/* Stage 2: Claim Extraction */}
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-neutral-400 mb-1.5 font-medium">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 flex items-center justify-center text-[10px] font-mono">2</span>
              <span>Claim Decomposition ({result.claims.length} {result.claims.length === 1 ? 'claim' : 'claims'})</span>
            </div>
            <div className="space-y-1.5">
              {result.claims.map((c, i) => (
                <div key={c.claim_id || i} className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800 flex items-center justify-between gap-3 text-[11px]">
                  <span className="text-slate-700 dark:text-zinc-300">
                    <strong className="text-slate-900 dark:text-white mr-1.5">Claim {i+1}:</strong> {c.claim_text}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-100 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 shrink-0">
                    {c.verdict.replace('_', ' ')} · {Math.round(c.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Stage 3: ML Linguistic Prior */}
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-neutral-400 mb-1.5 font-medium">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 flex items-center justify-center text-[10px] font-mono">3</span>
              <span>Linguistic Prior Assessment</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-[11px]">
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Stylistic Label</span>
                <span className="font-semibold text-slate-800 dark:text-zinc-200">{result.overall_verdict.ml_verdict || 'ANALYZED'}</span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">P(Disinformation | Text)</span>
                <span className="font-semibold text-slate-700 dark:text-zinc-300 font-mono">{(mlFakeProb * 100).toFixed(1)}%</span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">P(Authentic | Text)</span>
                <span className="font-semibold text-slate-700 dark:text-zinc-300 font-mono">{((1 - mlFakeProb) * 100).toFixed(1)}%</span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Role</span>
                <span className="font-medium text-slate-600 dark:text-neutral-400">Stylistic prior only</span>
              </div>
            </div>
          </div>

          {/* Stage 4: Retrieved Sources Matrix */}
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-neutral-400 mb-1.5 font-medium">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 flex items-center justify-center text-[10px] font-mono">4</span>
              <span>Evidence Evaluation ({result.all_evidence.length} sources evaluated)</span>
            </div>
            {result.all_evidence.length === 0 ? (
              <p className="text-slate-400 dark:text-neutral-500 italic text-[11px]">No external documents retrieved.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px] border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-neutral-800 text-slate-400 dark:text-neutral-500 font-medium">
                      <th className="py-1.5 pr-2">Publisher</th>
                      <th className="py-1.5 px-2">Authority</th>
                      <th className="py-1.5 px-2">Freshness</th>
                      <th className="py-1.5 px-2">Stance</th>
                      <th className="py-1.5 px-2">Independence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-neutral-800/60 text-slate-700 dark:text-zinc-300">
                    {result.all_evidence.map((ev, i) => (
                      <tr key={i} className="hover:bg-white dark:hover:bg-neutral-900/50">
                        <td className="py-1.5 pr-2 font-medium truncate max-w-[150px]">{ev.publisher || ev.domain || 'Source'}</td>
                        <td className="py-1.5 px-2 font-mono">{Math.round(ev.authority_score * 100)}%</td>
                        <td className="py-1.5 px-2 font-mono">{Math.round(ev.freshness_score * 100)}%</td>
                        <td className="py-1.5 px-2 font-medium">
                          {ev.relationship}
                        </td>
                        <td className="py-1.5 px-2 font-mono">
                          {ev.is_syndicated ? 'Syndicated' : 'Independent'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Stage 5: Decision Synthesis */}
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-neutral-400 mb-2 font-medium">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300 flex items-center justify-center text-[10px] font-mono">5</span>
              <span>Decision Synthesis</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-[11px] mb-2.5">
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Support</span>
                <span className="font-semibold text-emerald-700 dark:text-emerald-400 font-mono">{(supportScore * 100).toFixed(0)}%</span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Contradiction</span>
                <span className="font-semibold text-rose-700 dark:text-rose-400 font-mono">{(contraScore * 100).toFixed(0)}%</span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Conflicts</span>
                <span className="font-medium text-slate-700 dark:text-zinc-300">
                  {result.overall_verdict.has_conflicting_evidence ? 'Detected' : 'None'}
                </span>
              </div>
              <div className="p-2 rounded bg-white dark:bg-[#050505] border border-slate-200/60 dark:border-neutral-800">
                <span className="text-slate-400 dark:text-neutral-500 block text-[10px]">Verdict</span>
                <span className="font-semibold text-slate-900 dark:text-white font-mono">{result.overall_verdict.verdict}</span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-neutral-400 leading-relaxed">
              Synthesized by fusing linguistic prior with weighted evidence (Authority × Freshness × Independence × Stance Confidence). Verified refutations supersede stylistic scores.
            </p>
          </div>

          {/* Section 18: Expandable Analysis Details */}
          <div className="pt-2">
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="text-xs font-medium text-slate-500 hover:text-slate-900 dark:text-neutral-400 dark:hover:text-white flex items-center gap-1.5"
            >
              <span>{showTechnicalDetails ? 'Hide analysis details' : 'Show analysis details'}</span>
              {showTechnicalDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showTechnicalDetails && (
              <div className="mt-3 p-4 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800 text-[11px] text-slate-600 dark:text-zinc-300 space-y-2 leading-relaxed">
                <div>
                  <strong className="text-slate-900 dark:text-white">Model & Architecture:</strong> Linear SVM with TF-IDF sublinear word and character n-grams + NLI stance determination cross-referenced with Google Fact Check Tools API and primary registry records.
                </div>
                <div>
                  <strong className="text-slate-900 dark:text-white">Retrieval Strategy:</strong> Multi-query diversification (exact claim, entity-action, institutional domain filter, contradiction queries) to counteract confirmation bias in search engines.
                </div>
                <div>
                  <strong className="text-slate-900 dark:text-white">Evidence Weighting:</strong> Normalized score W_i = Authority × Freshness × Independence × Stance Confidence. Institutional domains (.gov, .edu, accredited registries) carry authority &gt; 0.90.
                </div>
                <div>
                  <strong className="text-slate-900 dark:text-white">Source Independence:</strong> Syndicated wire cluster detection groups duplicate reporting into a single origin so syndication reprints cannot inflate verification count.
                </div>
                <div>
                  <strong className="text-slate-900 dark:text-white">Confidence Methodology:</strong> Calibrated against the consistency, authority distribution, and stance agreement of retrieved external evidence.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
