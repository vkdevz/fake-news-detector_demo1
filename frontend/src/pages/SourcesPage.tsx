import React from 'react';
import { ShieldCheck } from 'lucide-react';

export const SourcesPage: React.FC = () => {
  const categories = [
    {
      title: 'Institutional & Official Registries',
      authority: '0.95 – 0.99',
      description: 'Primary agencies, peer-reviewed scientific repositories, space agencies, public health institutions, and statutory gazettes.',
      examples: ['ISRO, NASA, ESA', 'WHO, CDC, ICMR', '.gov & .edu authoritative registries']
    },
    {
      title: 'Accredited Fact-Checking Registries',
      authority: '0.90 – 0.95',
      description: 'Verified signatories adhering to non-partisan editorial standards, primary source cross-examinations, and transparent corrections.',
      examples: ['Google Fact Check Tools Registry', 'International Fact-Checking Network signatories', 'Accredited national verification desks']
    },
    {
      title: 'Independent Wire & News Reporting',
      authority: '0.80 – 0.90',
      description: 'Established editorial reporting with named correspondents, corroborating quotes, and multi-source verification standards.',
      examples: ['Reuters, Associated Press, AFP', 'National and international newsrooms with verified mastheads']
    },
    {
      title: 'Syndicated Wire Protection',
      authority: 'Cluster isolation',
      description: 'Automated clustering detects when multiple publishers reprint identical wire copy, grouping them into a single origin so syndication cannot artificially inflate evidence counts.',
      examples: ['Cross-domain entity matching', 'Lexical excerpt similarity', 'Single-origin evidence weighting']
    }
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8 animate-fadeIn">
      <div className="space-y-1.5 pb-5 border-b border-slate-200 dark:border-neutral-800">
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Source Standards & Authority
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-neutral-400">
          How TruthLens evaluates publishers, calculates independence, and isolates syndicated content.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {categories.map((cat, idx) => (
          <div 
            key={idx} 
            className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 space-y-3"
          >
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100">
                {cat.title}
              </h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-neutral-900 text-slate-700 dark:text-neutral-300 border border-slate-200/70 dark:border-neutral-800 shrink-0">
                {cat.authority}
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-neutral-400 leading-relaxed">
              {cat.description}
            </p>
            <div className="pt-2 border-t border-slate-100 dark:border-neutral-800/60">
              <span className="text-[10px] uppercase font-semibold text-slate-400 dark:text-neutral-500 block mb-1">
                Examples
              </span>
              <ul className="text-xs text-slate-700 dark:text-neutral-300 space-y-1">
                {cat.examples.map((ex, i) => (
                  <li key={i} className="flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-slate-400 dark:bg-neutral-600" />
                    <span>{ex}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 space-y-2">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          Network & SSRF Ingestion Defenses
        </h3>
        <p className="text-xs text-slate-600 dark:text-neutral-400 leading-relaxed">
          All external URL requests are evaluated through strict network guards that prevent access to private IP ranges, loopback addresses (127.0.0.1), and cloud metadata endpoints. Untrusted web text is sanitized to neutralize prompt-injection directives before evidence synthesis.
        </p>
      </div>
    </div>
  );
};
