import React from 'react';
import { EvidenceItem } from '../types/verification';

interface ConflictingAlertProps {
  evidences: EvidenceItem[];
  verdictLabel?: string;
}

export const ConflictingAlert: React.FC<ConflictingAlertProps> = ({ evidences, verdictLabel = 'PARTIALLY TRUE' }) => {
  const supporting = evidences.filter(e => e.relationship === 'SUPPORTS');
  const contradicting = evidences.filter(e => e.relationship === 'CONTRADICTS');

  if (supporting.length === 0 || contradicting.length === 0) {
    return null;
  }

  return (
    <div className="bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-800/40 rounded-xl p-5 sm:p-6 transition-colors">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-amber-900 dark:text-amber-200">
          Conflicting evidence
        </h3>
        <p className="text-xs text-amber-800/80 dark:text-amber-300/80 mt-0.5">
          Some sources support the claim while others contradict it.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
        {/* Supports Column */}
        <div className="bg-white/80 dark:bg-[#0c0c0e] border border-emerald-200/80 dark:border-emerald-800/40 rounded-lg p-3.5">
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 uppercase tracking-wide">
              Supports
            </span>
            <span className="text-xs font-mono font-medium text-emerald-600 dark:text-emerald-400">
              {supporting.length} {supporting.length === 1 ? 'source' : 'sources'}
            </span>
          </div>
          <div className="space-y-2">
            {supporting.slice(0, 3).map((item, idx) => (
              <div key={idx} className="text-xs">
                <span className="font-medium text-slate-800 dark:text-zinc-200 block truncate">
                  {item.publisher || item.domain || item.title}
                </span>
                <p className="text-[11px] text-slate-500 dark:text-neutral-400 italic line-clamp-2 mt-0.5">
                  "{item.excerpt}"
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Contradicts Column */}
        <div className="bg-white/80 dark:bg-[#0c0c0e] border border-rose-200/80 dark:border-rose-800/40 rounded-lg p-3.5">
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xs font-semibold text-rose-700 dark:text-rose-300 uppercase tracking-wide">
              Contradicts
            </span>
            <span className="text-xs font-mono font-medium text-rose-600 dark:text-rose-400">
              {contradicting.length} {contradicting.length === 1 ? 'source' : 'sources'}
            </span>
          </div>
          <div className="space-y-2">
            {contradicting.slice(0, 3).map((item, idx) => (
              <div key={idx} className="text-xs">
                <span className="font-medium text-slate-800 dark:text-zinc-200 block truncate">
                  {item.publisher || item.domain || item.title}
                </span>
                <p className="text-[11px] text-slate-500 dark:text-neutral-400 italic line-clamp-2 mt-0.5">
                  "{item.excerpt}"
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Assessment Column */}
        <div className="bg-white/80 dark:bg-[#0c0c0e] border border-slate-200/80 dark:border-neutral-800 rounded-lg p-3.5 flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-500 dark:text-neutral-400 uppercase tracking-wide block mb-1">
              Assessment
            </span>
            <span className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
              {verdictLabel.replace('_', ' ')}
            </span>
            <p className="text-xs text-slate-500 dark:text-neutral-400 mt-2 leading-relaxed">
              Disputed reports prevent an absolute determination without further corroboration.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
