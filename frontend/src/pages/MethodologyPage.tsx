import React from 'react';
import { BookOpen, Scale, ShieldCheck } from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Verification Methodology & Formulations
        </h2>
        <p className="text-sm sm:text-base text-slate-400">
          Theoretical formulation, mathematical models, scoring algorithms, and security defenses.
        </p>
      </div>

      {/* Verification Foundations */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-4">
        <div className="flex items-center gap-2 text-teal-400 font-mono text-xs font-bold uppercase">
          <BookOpen className="w-4 h-4" /> Core Architectural Foundation
        </div>
        <h3 className="text-xl font-bold text-white">
          Why Text Classification Alone Fails for Fake News Detection
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed">
          Traditional machine learning and deep learning approaches treat fake news as a closed-vocabulary sentiment or style classification problem (P(Fake | Text)). 
          However, deceptive articles written in formal, sober prose easily bypass stylistic classifiers, while legitimate urgent breaking news containing emotional words is falsely flagged. 
          TruthLens addresses this critical gap by introducing a <strong>hybrid evidence-grounded verification architecture</strong> that extracts atomic claims and validates them against an open web corpus.
        </p>
      </div>

      {/* Mathematical Formulations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ML Prior */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm space-y-3">
          <h4 className="text-base font-bold text-white">1. Linguistic Stylistic Prior (P_ML)</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            TF-IDF sublinear n-gram feature extraction coupled with a calibrated Linear SVM:
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-teal-300 overflow-x-auto">
            TF-IDF(t, d) = (1 + log(tf)) × log(|D| / df)
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Yields prior probability P(Fake | x), which serves as a stylistic prior but never overrides conflicting physical evidence.
          </p>
        </div>

        {/* Evidence Weighting */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm space-y-3">
          <h4 className="text-base font-bold text-white">2. Evidence Relevance & Authority (W_i)</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Each retrieved source i is scored across multiple verifiable dimensions:
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-teal-300 overflow-x-auto">
            W_i = Relevance_i × Authority_i × Freshness_i
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Where Authority is derived from institutional registries (.gov, .edu = 0.95+; accredited fact-checkers = 0.92; unknown blogs = 0.40).
          </p>
        </div>
      </div>

      {/* Hybrid Fusion Engine */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Scale className="w-5 h-5 text-teal-400" />
          3. Hybrid Fusion & 11 Granular Verdict Categories
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed">
          Unlike binary classifiers, TruthLens categorizes claims into 11 distinct real-world epistemic states:
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 text-xs font-mono">
          <span className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-center font-bold">SUPPORTED</span>
          <span className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-emerald-400 text-center font-bold">LIKELY_TRUE</span>
          <span className="p-2.5 rounded-lg bg-yellow-950/40 border border-yellow-500/40 text-yellow-300 text-center font-bold">PARTIALLY_TRUE</span>
          <span className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-500/40 text-amber-300 text-center font-bold">MISLEADING</span>
          <span className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 text-rose-400 text-center font-bold">LIKELY_FALSE</span>
          <span className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/40 text-rose-300 text-center font-bold">FALSE</span>
          <span className="p-2.5 rounded-lg bg-purple-950/40 border border-purple-500/40 text-purple-300 text-center font-bold">OUTDATED</span>
          <span className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-700 text-slate-400 text-center font-bold">UNVERIFIABLE</span>
          <span className="p-2.5 rounded-lg bg-cyan-950/40 border border-cyan-500/40 text-cyan-300 text-center font-bold">SATIRE</span>
          <span className="p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-500/30 text-cyan-400 text-center font-bold">OPINION</span>
          <span className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-center font-bold">UNSUPPORTED</span>
        </div>
      </div>

      {/* Security & Robustness */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-teal-400" />
          4. Security Architecture & SSRF Defenses
        </h3>
        <ul className="space-y-2 text-xs sm:text-sm text-slate-300 list-disc list-inside leading-relaxed">
          <li><strong>SSRF Prevention</strong>: Resolves hostnames and strictly blocks RFC1918 private subnets (127.0.0.1, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) and cloud metadata services (169.254.169.254).</li>
          <li><strong>Prompt-Injection Sanitization</strong>: Untrusted web text is filtered for adversarial directives (e.g. "Ignore previous instructions") before synthesis.</li>
          <li><strong>Deterministic Evidence Verification</strong>: TruthLens ensures LLMs cannot declare truthfulness arbitrarily; all verdicts map directly to tangible citations.</li>
        </ul>
      </div>
    </div>
  );
};
