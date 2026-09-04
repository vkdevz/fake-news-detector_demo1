import React from 'react';
import { VerificationResponse } from '../types/verification';
import { getVerdictTheme } from '../components/VerdictCard';

export interface VerificationHistoryItem {
  id: string;
  claim: string;
  verdict: string;
  confidence: number;
  timestamp: string;
  result: VerificationResponse;
}

interface HistoryPageProps {
  history: VerificationHistoryItem[];
  onSelectHistoryItem: (item: VerificationHistoryItem) => void;
  onClearHistory: () => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  history,
  onSelectHistoryItem,
  onClearHistory
}) => {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-6 animate-fadeIn">
      <div className="flex items-baseline justify-between pb-5 border-b border-slate-200 dark:border-neutral-800">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
            Verification History
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-neutral-400 mt-0.5">
            Previous claims evaluated during this session.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="text-xs text-slate-500 hover:text-slate-900 dark:text-neutral-400 dark:hover:text-white transition-colors"
          >
            Clear history
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="text-center py-16 bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-8 space-y-2">
          <p className="text-sm font-medium text-slate-700 dark:text-neutral-300">
            No verifications recorded
          </p>
          <p className="text-xs text-slate-400 dark:text-neutral-500">
            Claims and URLs verified during your session will be cataloged here for quick review.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {history.map((item) => {
            const theme = getVerdictTheme(item.verdict as any);
            return (
              <div
                key={item.id}
                onClick={() => onSelectHistoryItem(item)}
                className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 hover:border-slate-300 dark:hover:border-neutral-700 rounded-xl p-4 sm:p-5 cursor-pointer transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-900 dark:text-zinc-100 leading-snug">
                    "{item.claim}"
                  </p>
                  <span className="text-[11px] text-slate-400 dark:text-neutral-500 block font-mono">
                    {item.timestamp}
                  </span>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className={`px-2.5 py-1 text-xs font-semibold rounded border ${theme.badgeBg}`}>
                    {theme.label}
                  </span>
                  <span className="text-xs font-mono text-slate-500 dark:text-neutral-400">
                    {Math.round(item.confidence * 100)}% conf
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
