import React from 'react';
import { OverallVerdict, VerdictType } from '../types/verification';

interface VerdictCardProps {
  verdict: OverallVerdict;
  claimText: string;
  multipleClaims?: boolean;
}

export function getVerdictTheme(verdict: VerdictType) {
  switch (verdict) {
    case 'SUPPORTED':
    case 'LIKELY_TRUE':
      return {
        label: verdict === 'SUPPORTED' ? 'Supported' : 'Likely True',
        textColor: 'text-emerald-700 dark:text-emerald-400',
        badgeBg: 'bg-emerald-50 text-emerald-700 border-emerald-200/80 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800/40',
        barColor: 'bg-emerald-500',
        dotColor: 'bg-emerald-500',
      };
    case 'FALSE':
    case 'LIKELY_FALSE':
      return {
        label: verdict === 'FALSE' ? 'False' : 'Likely False',
        textColor: 'text-rose-700 dark:text-rose-400',
        badgeBg: 'bg-rose-50 text-rose-700 border-rose-200/80 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-800/40',
        barColor: 'bg-rose-500',
        dotColor: 'bg-rose-500',
      };
    case 'MISLEADING':
      return {
        label: 'Misleading',
        textColor: 'text-amber-700 dark:text-amber-400',
        badgeBg: 'bg-amber-50 text-amber-700 border-amber-200/80 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800/40',
        barColor: 'bg-amber-500',
        dotColor: 'bg-amber-500',
      };
    case 'OUTDATED':
      return {
        label: 'Outdated',
        textColor: 'text-orange-700 dark:text-orange-400',
        badgeBg: 'bg-orange-50 text-orange-700 border-orange-200/80 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800/40',
        barColor: 'bg-orange-500',
        dotColor: 'bg-orange-500',
      };
    case 'PARTIALLY_TRUE':
      return {
        label: 'Partially True',
        textColor: 'text-sky-700 dark:text-sky-400',
        badgeBg: 'bg-sky-50 text-sky-700 border-sky-200/80 dark:bg-sky-950/40 dark:text-sky-300 dark:border-sky-800/40',
        barColor: 'bg-sky-500',
        dotColor: 'bg-sky-500',
      };
    case 'SATIRE':
    case 'OPINION':
      return {
        label: verdict === 'SATIRE' ? 'Satire' : 'Opinion',
        textColor: 'text-slate-700 dark:text-neutral-300',
        badgeBg: 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-neutral-900 dark:text-neutral-300 dark:border-neutral-800',
        barColor: 'bg-slate-500',
        dotColor: 'bg-neutral-400',
      };
    case 'UNVERIFIABLE':
    case 'UNSUPPORTED':
    default:
      return {
        label: 'Unverifiable',
        textColor: 'text-slate-700 dark:text-neutral-300',
        badgeBg: 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-neutral-900 dark:text-neutral-300 dark:border-neutral-800',
        barColor: 'bg-neutral-500',
        dotColor: 'bg-neutral-400',
      };
  }
}

export const VerdictCard: React.FC<VerdictCardProps> = ({ verdict, claimText, multipleClaims }) => {
  const theme = getVerdictTheme(verdict.verdict);
  const confPercent = Math.round(verdict.confidence * 100);

  return (
    <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-6 sm:p-8 transition-colors">
      {/* Top Header: Verdict & Confidence */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 pb-5 border-b border-slate-100 dark:border-neutral-800">
        <div>
          <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-500 block mb-1">
            Verdict
          </span>
          <div className="flex items-center gap-3">
            <span className={`text-2xl sm:text-3xl font-semibold tracking-tight ${theme.textColor}`}>
              {theme.label.toUpperCase()}
            </span>
            {verdict.has_conflicting_evidence && (
              <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800/50">
                Conflicting evidence
              </span>
            )}
            {verdict.is_outdated && (
              <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-orange-50 text-orange-700 border border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800/50">
                Outdated
              </span>
            )}
          </div>
        </div>

        {/* Confidence Display */}
        <div className="sm:text-right min-w-[180px]">
          <div className="flex sm:justify-end items-baseline gap-1.5">
            <span className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
              {confPercent}%
            </span>
            <span className="text-xs text-slate-500 dark:text-neutral-400">
              confidence
            </span>
          </div>

          {/* Subtle horizontal indicator */}
          <div className="w-full sm:w-36 bg-slate-100 dark:bg-neutral-800 h-1 rounded-full mt-1.5 overflow-hidden ml-auto">
            <div 
              className={`h-full ${theme.barColor} transition-all duration-300`} 
              style={{ width: `${confPercent}%` }} 
            />
          </div>
          <span className="text-[11px] text-slate-400 dark:text-neutral-500 block mt-1">
            Reflects consistency & quality of evidence
          </span>
        </div>
      </div>

      {/* Claim Section */}
      <div className="py-4 border-b border-slate-100 dark:border-neutral-800">
        <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-500 block mb-1">
          {multipleClaims ? 'Primary claim analyzed' : 'Claim'}
        </span>
        <p className="text-base sm:text-lg font-medium text-slate-900 dark:text-zinc-100 leading-snug">
          "{claimText}"
        </p>
      </div>

      {/* Grounded Explanation */}
      <div className="pt-4">
        <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-500 block mb-1.5">
          Assessment
        </span>
        <div className="text-sm text-slate-600 dark:text-zinc-300 leading-relaxed font-sans space-y-2">
          {verdict.explanation
            .replace(/VerdictEnum\./g, '')
            .split('\n\n')
            .map((paragraph, pIdx) => {
              // Parse basic bold markdown
              const parts = paragraph.split(/(\*\*[^*]+\*\*)/g);
              return (
                <p key={pIdx}>
                  {parts.map((part, idx) => {
                    if (part.startsWith('**') && part.endsWith('**')) {
                      return (
                        <strong key={idx} className="font-semibold text-slate-900 dark:text-zinc-100">
                          {part.slice(2, -2)}
                        </strong>
                      );
                    }
                    return part;
                  })}
                </p>
              );
            })}
        </div>
      </div>
    </div>
  );
};
