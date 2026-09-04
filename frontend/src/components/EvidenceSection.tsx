import React, { useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { EvidenceItem } from '../types/verification';

interface EvidenceSectionProps {
  evidences: EvidenceItem[];
  selectedClaimId: string | null;
}

export const EvidenceSection: React.FC<EvidenceSectionProps> = ({ evidences, selectedClaimId }) => {
  const [activeFilter, setActiveFilter] = useState<'ALL' | 'SUPPORTS' | 'CONTRADICTS' | 'CONTEXT'>('ALL');

  // Filter by selected claim if user clicked a claim in ClaimBreakdown
  const claimFiltered = selectedClaimId
    ? evidences.filter((e) => e.claim_id === selectedClaimId)
    : evidences;

  // Filter by stance
  const filtered = claimFiltered.filter((e) => {
    if (activeFilter === 'ALL') return true;
    if (activeFilter === 'SUPPORTS') return e.relationship === 'SUPPORTS';
    if (activeFilter === 'CONTRADICTS') return e.relationship === 'CONTRADICTS';
    if (activeFilter === 'CONTEXT') return ['PARTIALLY_SUPPORTS', 'NEUTRAL', 'OUTDATED'].includes(e.relationship);
    return true;
  });

  // Calculate summary metrics
  const totalReviewed = claimFiltered.length;
  const supportingCount = claimFiltered.filter((e) => e.relationship === 'SUPPORTS').length;
  const conflictingCount = claimFiltered.filter((e) => e.relationship === 'CONTRADICTS').length;
  const uniqueOrigins = new Set(claimFiltered.map((e) => e.cluster_id || e.url)).size;

  const getSourceTypeLabel = (sourceType: string, domain?: string) => {
    const d = (domain || '').toLowerCase();
    if (d.endsWith('.gov') || d.endsWith('.mil') || d.includes('isro.') || d.includes('nasa.')) {
      return 'Official source';
    }
    if (sourceType === 'fact_check' || d.includes('factcheck') || d.includes('snopes') || d.includes('politifact')) {
      return 'Fact-checking registry';
    }
    if (d.includes('reuters') || d.includes('apnews') || d.includes('bbc') || d.includes('nytimes') || d.includes('thehindu')) {
      return 'Independent reporting';
    }
    return 'Secondary source';
  };

  const getRelationshipBadge = (rel: string) => {
    switch (rel) {
      case 'SUPPORTS':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/80 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800/40">
            Supports
          </span>
        );
      case 'CONTRADICTS':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-200/80 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-800/40">
            Contradicts
          </span>
        );
      case 'PARTIALLY_SUPPORTS':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200/80 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800/40">
            Partially supports
          </span>
        );
      case 'OUTDATED':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-orange-50 text-orange-700 border border-orange-200/80 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800/40">
            Outdated context
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200 dark:bg-neutral-900 dark:text-neutral-300 dark:border-neutral-800">
            Contextual
          </span>
        );
    }
  };

  return (
    <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-6 sm:p-8 transition-colors">
      {/* Evidence Summary Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 pb-4 border-b border-slate-100 dark:border-neutral-800">
        <div>
          <span className="text-[11px] uppercase tracking-widest font-semibold text-slate-400 dark:text-neutral-500 block mb-1">
            Evidence
          </span>
          <div className="flex items-center flex-wrap gap-2 text-xs sm:text-sm text-slate-700 dark:text-zinc-300 font-medium">
            <span>{totalReviewed} {totalReviewed === 1 ? 'source' : 'sources'} reviewed</span>
            <span className="text-slate-300 dark:text-neutral-700">·</span>
            <span className="text-emerald-700 dark:text-emerald-400 font-semibold">{supportingCount} supporting</span>
            {conflictingCount > 0 && (
              <>
                <span className="text-slate-300 dark:text-neutral-700">·</span>
                <span className="text-rose-700 dark:text-rose-400 font-semibold">{conflictingCount} conflicting</span>
              </>
            )}
            <span className="text-slate-300 dark:text-neutral-700">·</span>
            <span>{uniqueOrigins} independent {uniqueOrigins === 1 ? 'origin' : 'origins'}</span>
          </div>
        </div>

        {/* Filter buttons */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-neutral-900 p-0.5 rounded-lg border border-slate-200 dark:border-neutral-800 self-start sm:self-auto text-xs">
          {(['ALL', 'SUPPORTS', 'CONTRADICTS', 'CONTEXT'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveFilter(tab)}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeFilter === tab
                  ? 'bg-white dark:bg-neutral-800 text-slate-900 dark:text-white font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-800 dark:text-neutral-400 dark:hover:text-white'
              }`}
            >
              {tab === 'ALL' ? 'All' : tab.charAt(0) + tab.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Sources List */}
      {filtered.length === 0 ? (
        <div className="text-center py-10 text-slate-400 dark:text-neutral-500 text-xs">
          No sources match the selected filter.
        </div>
      ) : (
        <div className="divide-y divide-slate-100 dark:divide-neutral-800 mt-2">
          {filtered.map((item, idx) => {
            const sourceTypeLabel = getSourceTypeLabel(item.source_type, item.domain);

            return (
              <div key={item.id || idx} className="py-4 first:pt-3 last:pb-0">
                <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 mb-1.5">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
                      {item.publisher || item.domain || 'Primary Source'}
                    </span>
                    <span className="text-xs text-slate-400 dark:text-neutral-500">
                      {sourceTypeLabel}
                    </span>
                    {item.publication_date && (
                      <>
                        <span className="text-slate-300 dark:text-neutral-700">·</span>
                        <span className="text-xs text-slate-400 dark:text-neutral-500">
                          {item.publication_date}
                        </span>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {getRelationshipBadge(item.relationship)}
                  </div>
                </div>

                {/* Syndicated Source Notice */}
                {item.is_syndicated && (
                  <div className="my-2 px-2.5 py-1 rounded bg-amber-50/80 dark:bg-amber-950/30 border border-amber-200/60 dark:border-amber-800/40 text-[11px] text-amber-800 dark:text-amber-300">
                    <strong className="font-semibold">Syndicated source:</strong> This report appears to originate from the same underlying wire source as another result.
                  </div>
                )}

                {/* Article Title & Link */}
                <h4 className="text-xs sm:text-sm font-medium text-slate-800 dark:text-zinc-200 hover:text-teal-600 dark:hover:text-teal-400 transition-colors my-1">
                  <a 
                    href={item.url} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="inline-flex items-center gap-1.5"
                  >
                    <span>{item.title}</span>
                    <ExternalLink className="w-3 h-3 text-slate-400 dark:text-neutral-500 shrink-0" />
                  </a>
                </h4>

                {/* Direct Excerpt */}
                <p className="text-xs text-slate-600 dark:text-zinc-300 leading-relaxed mt-1 font-sans pl-3 border-l border-slate-200 dark:border-neutral-800 italic">
                  "{item.excerpt}"
                </p>

                {/* Assessment note if present */}
                {item.reasoning && (
                  <p className="text-[11px] text-slate-400 dark:text-neutral-500 mt-1 pl-3">
                    {item.reasoning}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
