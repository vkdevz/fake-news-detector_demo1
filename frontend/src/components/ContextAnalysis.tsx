import React from 'react';
import { TimelineEvent } from '../types/verification';

interface ContextAnalysisProps {
  claimText: string;
  isOutdated?: boolean;
  isMisleading?: boolean;
  events?: TimelineEvent[];
}

export const ContextAnalysis: React.FC<ContextAnalysisProps> = ({
  claimText,
  isOutdated,
  isMisleading,
  events,
}) => {
  if (!isOutdated && !isMisleading && (!events || events.length === 0)) {
    return null;
  }

  // Detect statistical percentage change in claim if misleading
  const statMatch = claimText.match(/(\d+(?:\.\d+)?%|\d+\s*percent)/i);
  const claimedStat = statMatch ? statMatch[0] : '+200%';

  return (
    <div className="space-y-4">
      {/* Temporal Context Section (Section 15) */}
      {isOutdated && (
        <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 sm:p-6 transition-colors">
          <div className="mb-3">
            <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-400 block mb-0.5">
              Time context
            </span>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
              Chronologically superseded assertion
            </h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1 text-xs">
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Claim</span>
              <p className="text-slate-800 dark:text-zinc-200 font-medium">"{claimText}"</p>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Assessment</span>
              <span className="inline-block font-semibold text-orange-700 dark:text-orange-400 text-sm">
                Outdated
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Why</span>
              <p className="text-slate-600 dark:text-neutral-300">
                The claim was once accurate during a prior timeframe, but has been superseded by subsequent events.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Misleading Statistics Section (Section 16) */}
      {isMisleading && (
        <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 sm:p-6 transition-colors">
          <div className="mb-3">
            <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-400 block mb-0.5">
              Context
            </span>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
              Statistical framing without baseline context
            </h3>
            <p className="text-xs text-slate-500 dark:text-neutral-400 mt-1">
              The reported figure cannot be interpreted accurately without an established baseline or denominator.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1 text-xs">
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Claimed change</span>
              <span className="text-sm font-bold text-slate-900 dark:text-zinc-100 font-mono">
                {claimedStat}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Missing context</span>
              <span className="text-sm font-semibold text-amber-700 dark:text-amber-400">
                Baseline data & sample size
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-black/60 border border-slate-200/70 dark:border-neutral-800">
              <span className="text-slate-400 dark:text-neutral-400 block text-[11px] font-medium mb-1">Assessment</span>
              <span className="text-sm font-bold text-amber-700 dark:text-amber-400">
                MISLEADING
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Chronological Timeline Events (if available) */}
      {events && events.length > 0 && (
        <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 sm:p-6 transition-colors">
          <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-400 block mb-3">
            Timeline
          </span>

          <div className="space-y-3 pl-2 border-l border-slate-200 dark:border-neutral-800 ml-1">
            {events.map((evt, idx) => (
              <div key={idx} className="relative pl-4">
                <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-slate-400 dark:bg-neutral-600 border-2 border-white dark:border-black" />
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-800 dark:text-zinc-200">
                    {evt.title}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400 shrink-0">
                    {evt.date}
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-neutral-400 mt-0.5">
                  {evt.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
