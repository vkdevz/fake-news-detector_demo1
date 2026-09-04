import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';
import { OverallVerdict } from '../types/verification';

interface ModelComparisonCardProps {
  verdict: OverallVerdict;
}

export const ModelComparisonCard: React.FC<ModelComparisonCardProps> = ({ verdict }) => {
  const mlFakeProb = verdict.ml_probability !== undefined ? Math.round(verdict.ml_probability * 100) : 50;
  const mlRealProb = 100 - mlFakeProb;
  const mlVerdict = verdict.ml_verdict || (mlFakeProb > 50 ? 'FAKE' : 'REAL');

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-3">
        <Cpu className="w-5 h-5 text-teal-400" />
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
          Architectural Comparison: Classical ML vs. Evidence-Driven Hybrid
        </h3>
      </div>

      <p className="text-xs text-slate-400 mb-6 leading-relaxed">
        Contrasts stylistic classification against multi-source evidence verification: while traditional ML classifiers inspect only stylistic vocabulary, TruthLens cross-examines factual claims against external primary evidence.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Classical ML Column */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono font-bold text-slate-400 uppercase">
              1. Classical ML Classifier
            </span>
            <span className={`text-xs px-2 py-0.5 rounded font-bold ${
              mlVerdict === 'REAL' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
            }`}>
              {mlVerdict}
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-3">
            Based on TF-IDF n-grams, sensational word frequency, and stylistic sentiment tokens.
          </p>
          <div className="space-y-1.5 text-xs font-mono">
            <div className="flex justify-between text-slate-400 text-[11px]">
              <span>Fake Probability: {mlFakeProb}%</span>
              <span>Real Probability: {mlRealProb}%</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden flex">
              <div className="bg-rose-500 h-full" style={{ width: `${mlFakeProb}%` }} />
              <div className="bg-emerald-500 h-full" style={{ width: `${mlRealProb}%` }} />
            </div>
          </div>
        </div>

        {/* TruthLens Hybrid Determination */}
        <div className="bg-teal-950/20 border border-teal-500/40 rounded-xl p-5 relative overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono font-bold text-teal-300 uppercase flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-teal-400" />
              2. TruthLens Hybrid Verdict
            </span>
            <span className="text-xs px-2 py-0.5 rounded font-bold bg-teal-500/20 text-teal-300 border border-teal-500/30">
              {verdict.verdict.replace('_', ' ')}
            </span>
          </div>
          <p className="text-xs text-slate-300 mb-3 leading-relaxed">
            Fuses ML linguistic priors with multi-source retrieval, NLI contradiction analysis, and temporal validation.
          </p>
          <div className="space-y-1.5 text-xs font-mono">
            <div className="flex justify-between text-slate-300 text-[11px]">
              <span>Contradiction: {Math.round(verdict.contradiction_score * 100)}%</span>
              <span>Support: {Math.round(verdict.support_score * 100)}%</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden flex">
              <div className="bg-rose-500 h-full" style={{ width: `${Math.round(verdict.contradiction_score * 100)}%` }} />
              <div className="bg-emerald-500 h-full" style={{ width: `${Math.round(verdict.support_score * 100)}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
