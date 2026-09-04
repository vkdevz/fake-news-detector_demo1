import React from 'react';
import { ClaimVerdict } from '../types/verification';
import { getVerdictTheme } from './VerdictCard';

interface ClaimBreakdownProps {
  claims: ClaimVerdict[];
  selectedClaimId: string | null;
  onSelectClaim: (claimId: string | null) => void;
}

export const ClaimBreakdown: React.FC<ClaimBreakdownProps> = ({
  claims,
  selectedClaimId,
  onSelectClaim
}) => {
  if (!claims || claims.length <= 1) {
    return null; // For single claims, it's displayed directly inside VerdictCard
  }

  return (
    <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-6 transition-colors">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-neutral-800 mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
            Claims analyzed ({claims.length})
          </h3>
          <p className="text-xs text-slate-500 dark:text-neutral-400 mt-0.5">
            Select a claim to filter corroborating and contradicting evidence below.
          </p>
        </div>
        {selectedClaimId && (
          <button
            onClick={() => onSelectClaim(null)}
            className="text-xs font-medium text-slate-600 dark:text-neutral-300 hover:text-slate-900 dark:hover:text-white underline underline-offset-2"
          >
            Show all sources
          </button>
        )}
      </div>

      <div className="space-y-2.5">
        {claims.map((claim, idx) => {
          const theme = getVerdictTheme(claim.verdict);
          const isSelected = selectedClaimId === claim.claim_id;

          return (
            <div
              key={claim.claim_id || idx}
              onClick={() => onSelectClaim(isSelected ? null : claim.claim_id)}
              className={`p-3.5 rounded-lg border transition-all cursor-pointer ${
                isSelected
                  ? 'border-slate-400 dark:border-neutral-600 bg-slate-50 dark:bg-black/60'
                  : 'border-slate-200/80 dark:border-neutral-800 hover:border-slate-300 dark:hover:border-neutral-700 bg-transparent'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-slate-400 dark:text-neutral-500">
                      #{idx + 1}
                    </span>
                    <p className="text-sm font-medium text-slate-900 dark:text-zinc-100 leading-snug">
                      "{claim.claim_text}"
                    </p>
                  </div>
                  {claim.reason && (
                    <p className="text-xs text-slate-500 dark:text-neutral-400 pl-5">
                      {claim.reason}
                    </p>
                  )}
                </div>

                {/* Verdict Badge */}
                <div className="shrink-0 text-right">
                  <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded border ${theme.badgeBg}`}>
                    {theme.label}
                  </span>
                  <span className="block text-[11px] text-slate-400 dark:text-neutral-500 mt-0.5 font-mono">
                    {Math.round(claim.confidence * 100)}% conf
                  </span>
                </div>
              </div>

              {/* Source counts */}
              <div className="flex items-center gap-3 mt-2 pt-2 border-t border-slate-100 dark:border-neutral-800/60 text-[11px] text-slate-500 dark:text-neutral-400 pl-5">
                <span className="text-emerald-600 dark:text-emerald-400">
                  {claim.supporting_evidence_count} supporting
                </span>
                <span>·</span>
                <span className="text-rose-600 dark:text-rose-400">
                  {claim.contradicting_evidence_count} contradicting
                </span>
                {isSelected && (
                  <>
                    <span>·</span>
                    <span className="text-slate-900 dark:text-white font-medium">
                      Filtering evidence
                    </span>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
