import React from 'react';
import { ShieldCheck, Code2, AlertTriangle } from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      <div className="text-center space-y-3">
        <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          About TruthLens
        </h2>
        <p className="text-sm sm:text-base text-slate-500 dark:text-slate-400">
          An evidence-grounded verification system for claims, statements, and sources.
        </p>
      </div>

      <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200 dark:border-neutral-800 rounded-2xl p-6 sm:p-8 space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-neutral-900 text-slate-900 dark:text-white flex items-center justify-center border border-slate-200/70 dark:border-neutral-800">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">TruthLens Verification System</h3>
            <p className="text-xs text-slate-500 dark:text-neutral-400">Evidence before belief</p>
          </div>
        </div>

        <div className="space-y-3 text-sm text-slate-600 dark:text-neutral-300 leading-relaxed">
          <p>
            <strong>Problem Statement:</strong> The dissemination of digital disinformation poses serious challenges to social stability, democratic institutions, and public trust. Existing automated approaches predominantly rely on shallow linguistic classifiers that inspect only vocabulary without validating whether assertions reflect real-world facts.
          </p>
          <p>
            <strong>System Architecture:</strong> TruthLens provides a hybrid, evidence-grounded verification framework. The system decomposes incoming content into atomic propositions, retrieves corroborated documentation across accredited primary and secondary sources, evaluates publisher authority, detects contradictions, accounts for temporal obsolescence, and produces traceable explanations.
          </p>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200 dark:border-neutral-800 rounded-2xl p-6 sm:p-8 space-y-4">
        <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Code2 className="w-5 h-5 text-slate-700 dark:text-neutral-300" />
          Core Engineering & Technology Stack
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="bg-slate-50 dark:bg-black p-4 rounded-xl border border-slate-200 dark:border-neutral-800 space-y-1">
            <span className="font-bold text-slate-900 dark:text-white font-mono block">Backend & Inference</span>
            <p className="text-slate-600 dark:text-neutral-400">Python 3.12, FastAPI, Scikit-Learn, NumPy, Pandas, Pydantic v2, BeautifulSoup4, HTTPX</p>
          </div>
          <div className="bg-slate-50 dark:bg-black p-4 rounded-xl border border-slate-200 dark:border-neutral-800 space-y-1">
            <span className="font-bold text-slate-900 dark:text-white font-mono block">Frontend & Interface</span>
            <p className="text-slate-600 dark:text-neutral-400">React 18, TypeScript, Vite, Tailwind CSS, Lucide React, Minimal Editorial Theme</p>
          </div>
          <div className="bg-slate-50 dark:bg-black p-4 rounded-xl border border-slate-200 dark:border-neutral-800 space-y-1">
            <span className="font-bold text-slate-900 dark:text-white font-mono block">Persistence & Audit</span>
            <p className="text-slate-600 dark:text-neutral-400">SQLite with SQLAlchemy 2.0 (Relational schema: 10 traceable audit tables)</p>
          </div>
          <div className="bg-slate-50 dark:bg-black p-4 rounded-xl border border-slate-200 dark:border-neutral-800 space-y-1">
            <span className="font-bold text-slate-900 dark:text-white font-mono block">Verification & Security</span>
            <p className="text-slate-600 dark:text-neutral-400">Google Fact Check Tools API, SSRF Network Guards, Multi-Query Generator, NLI Stance Classifier</p>
          </div>
        </div>
      </div>

      {/* Epistemic Disclaimer */}
      <div className="p-6 rounded-2xl bg-slate-100 dark:bg-[#0c0c0e] border border-slate-200 dark:border-neutral-800 text-xs sm:text-sm text-slate-600 dark:text-neutral-300 space-y-2">
        <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white">
          <AlertTriangle className="w-4 h-4 text-amber-500" />
          Verification Scope & Epistemic Notice
        </div>
        <p className="leading-relaxed text-slate-500 dark:text-neutral-400">
          TruthLens assesses whether a claim is substantiated or contradicted by publicly available evidence and accredited reporting. It evaluates statements strictly against the state of public knowledge, retrieved documentation, and calibrated heuristics.
        </p>
      </div>
    </div>
  );
};
