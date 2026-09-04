import React, { useState, useEffect } from 'react';
import { 
  RefreshCw, AlertCircle, CheckCircle2, Circle
} from 'lucide-react';
import { verifyText, verifyUrl, verifyClaim, getDemoSamples } from '../services/api';
import { VerificationResponse, DemoScenario } from '../types/verification';
import { VerdictCard } from '../components/VerdictCard';
import { ClaimBreakdown } from '../components/ClaimBreakdown';
import { EvidenceSection } from '../components/EvidenceSection';
import { ConflictingAlert } from '../components/ConflictingAlert';
import { ContextAnalysis } from '../components/ContextAnalysis';
import { VerificationTrace } from '../components/VerificationTrace';

interface VerifyPageProps {
  onVerificationComplete?: (result: VerificationResponse) => void;
  externalResult?: VerificationResponse | null;
}

export const VerifyPage: React.FC<VerifyPageProps> = ({ 
  onVerificationComplete,
  externalResult 
}) => {
  const [inputMode, setInputMode] = useState<'claim' | 'url' | 'text'>('claim');
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<VerificationResponse | null>(null);
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [, setServerSamples] = useState<DemoScenario[]>([]);

  // Default suggested examples matching Section 8
  const defaultExamples = [
    {
      title: 'ESA Euclid Launch',
      text: 'The European Space Agency launched Euclid in July 2023.',
      mode: 'claim' as const,
    },
    {
      title: 'Mars Highways Claim',
      text: 'NASA photographed paved highways on Mars.',
      mode: 'claim' as const,
    },
    {
      title: 'British Monarchy Status',
      text: 'Queen Elizabeth II is the reigning monarch of the United Kingdom.',
      mode: 'claim' as const,
    },
    {
      title: 'Statistical Surge',
      text: 'Municipal police reports confirmed that regional violent crime increased 200 percent this year.',
      mode: 'claim' as const,
    },
    {
      title: 'ISRO Aditya-L1 Mission',
      text: 'ISRO successfully launched the Aditya-L1 spacecraft to study the Sun from Lagrange point L1.',
      mode: 'claim' as const,
    },
  ];

  useEffect(() => {
    getDemoSamples()
      .then(setServerSamples)
      .catch((err) => console.error('Error fetching sample claims:', err));
  }, []);

  // Synchronize when externalResult is selected from History
  useEffect(() => {
    if (externalResult) {
      setResult(externalResult);
      setInputValue(externalResult.raw_input);
      setInputMode((externalResult.input_type as any) || 'claim');
    }
  }, [externalResult]);

  const handleVerify = async () => {
    if (!inputValue.trim()) {
      setErrorMessage('Please enter a claim, statement, or URL to verify.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setSelectedClaimId(null);
    setLoadingStage(1);

    // Minimal stage progression reflecting genuine verification steps
    const stageTimer1 = setTimeout(() => setLoadingStage(2), 350);
    const stageTimer2 = setTimeout(() => setLoadingStage(3), 850);
    const stageTimer3 = setTimeout(() => setLoadingStage(4), 1400);

    try {
      let res: VerificationResponse;
      if (inputMode === 'url') {
        res = await verifyUrl(inputValue.trim());
      } else if (inputMode === 'text') {
        res = await verifyText(inputValue.trim());
      } else {
        res = await verifyClaim(inputValue.trim());
      }

      setResult(res);
      if (onVerificationComplete) {
        onVerificationComplete(res);
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Live evidence could not be retrieved. Try again or check the input claim.');
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      setIsLoading(false);
      setLoadingStage(0);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (inputMode === 'url' || inputMode === 'claim') {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleVerify();
      }
    } else if (inputMode === 'text') {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleVerify();
      }
    }
  };

  const handleSelectExample = (text: string, mode: 'claim' | 'url' | 'text' = 'claim') => {
    setInputMode(mode);
    setInputValue(text);
    setResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8 animate-fadeIn">
      {/* Hero Section (Section 6 & 20) */}
      <div className="space-y-1.5">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Verify a claim
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
          Examine a claim against available evidence, source quality, context, and time.
        </p>
      </div>

      {/* Primary Input Container (Section 7) */}
      <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800/90 rounded-xl p-4 sm:p-5 shadow-xs transition-colors space-y-3">
        {/* Subtle segmented mode controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setInputMode('claim')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              inputMode === 'claim'
                ? 'bg-slate-100 dark:bg-neutral-800 text-slate-900 dark:text-white font-semibold'
                : 'text-slate-500 dark:text-neutral-400 hover:text-slate-800 dark:hover:text-zinc-200'
            }`}
          >
            Claim
          </button>
          <button
            onClick={() => setInputMode('url')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              inputMode === 'url'
                ? 'bg-slate-100 dark:bg-neutral-800 text-slate-900 dark:text-white font-semibold'
                : 'text-slate-500 dark:text-neutral-400 hover:text-slate-800 dark:hover:text-zinc-200'
            }`}
          >
            URL
          </button>
          <button
            onClick={() => setInputMode('text')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              inputMode === 'text'
                ? 'bg-slate-100 dark:bg-neutral-800 text-slate-900 dark:text-white font-semibold'
                : 'text-slate-500 dark:text-neutral-400 hover:text-slate-800 dark:hover:text-zinc-200'
            }`}
          >
            Text
          </button>
        </div>

        {/* Input area */}
        <div className="relative">
          {inputMode === 'url' ? (
            <input
              type="url"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter an article URL to verify…"
              className="w-full px-3.5 py-3 rounded-lg bg-slate-50 dark:bg-black border border-slate-200 dark:border-neutral-800 text-slate-900 dark:text-zinc-100 placeholder-slate-400 dark:placeholder-neutral-500 text-sm font-sans focus:outline-none focus:border-slate-400 dark:focus:border-neutral-600 transition-colors"
            />
          ) : (
            <textarea
              rows={inputMode === 'text' ? 5 : 2}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                inputMode === 'text'
                  ? 'Paste complete article or statements to decompose into claims…'
                  : 'Enter a claim, statement, or URL…'
              }
              className="w-full px-3.5 py-3 rounded-lg bg-slate-50 dark:bg-black border border-slate-200 dark:border-neutral-800 text-slate-900 dark:text-zinc-100 placeholder-slate-400 dark:placeholder-neutral-500 text-sm font-sans leading-relaxed resize-none focus:outline-none focus:border-slate-400 dark:focus:border-neutral-600 transition-colors"
            />
          )}
        </div>

        {/* Bottom Actions Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
          <span className="text-[11px] text-slate-400 dark:text-neutral-500">
            {inputMode === 'text' ? 'Press ⌘+Enter to verify' : 'Press Enter to verify'}
          </span>

          <button
            onClick={handleVerify}
            disabled={isLoading || !inputValue.trim()}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-slate-900 text-white dark:bg-white dark:text-black text-xs font-semibold hover:bg-slate-800 dark:hover:bg-neutral-200 disabled:opacity-40 disabled:cursor-not-allowed transition-all self-end sm:self-auto"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Verifying…</span>
              </>
            ) : (
              <span>Verify</span>
            )}
          </button>
        </div>
      </div>

      {/* Subtle Example Claims (Section 8 & 20) */}
      {!result && !isLoading && (
        <div className="space-y-2 pt-1">
          <span className="text-xs font-medium text-slate-400 dark:text-neutral-500 block">
            Try an example
          </span>
          <div className="flex flex-wrap gap-2">
            {defaultExamples.map((ex, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectExample(ex.text, ex.mode)}
                className="text-left text-xs px-3 py-1.5 rounded-lg bg-white dark:bg-[#0c0c0e] border border-slate-200/80 dark:border-neutral-800 text-slate-700 dark:text-zinc-300 hover:text-slate-950 dark:hover:text-white hover:border-slate-300 dark:hover:border-neutral-700 transition-colors"
              >
                "{ex.text}"
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading State (Section 21) */}
      {isLoading && (
        <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-6 space-y-4 animate-fadeIn">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-slate-600 dark:text-neutral-400" />
            <span className="text-sm font-semibold text-slate-900 dark:text-white">
              Analyzing claim
            </span>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center gap-2.5 text-slate-700 dark:text-zinc-300">
              {loadingStage >= 1 ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <Circle className="w-3.5 h-3.5 text-slate-300 dark:text-neutral-700" />
              )}
              <span className={loadingStage === 1 ? 'font-semibold text-slate-900 dark:text-white' : ''}>
                Claim decomposition
              </span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-700 dark:text-zinc-300">
              {loadingStage >= 2 ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <Circle className="w-3.5 h-3.5 text-slate-300 dark:text-neutral-700" />
              )}
              <span className={loadingStage === 2 ? 'font-semibold text-slate-900 dark:text-white' : ''}>
                Evidence retrieval
              </span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-700 dark:text-zinc-300">
              {loadingStage >= 3 ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <Circle className="w-3.5 h-3.5 text-slate-300 dark:text-neutral-700" />
              )}
              <span className={loadingStage === 3 ? 'font-semibold text-slate-900 dark:text-white' : ''}>
                Source analysis
              </span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-700 dark:text-zinc-300">
              {loadingStage >= 4 ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <Circle className="w-3.5 h-3.5 text-slate-300 dark:text-neutral-700" />
              )}
              <span className={loadingStage === 4 ? 'font-semibold text-slate-900 dark:text-white' : ''}>
                Context verification
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Error State (Section 22) */}
      {errorMessage && (
        <div className="bg-white dark:bg-[#0c0c0e] border border-slate-200/90 dark:border-neutral-800 rounded-xl p-5 space-y-2 animate-fadeIn">
          <div className="flex items-center gap-2 text-slate-900 dark:text-white font-semibold text-sm">
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
            <span>Verification unavailable</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-neutral-400 leading-relaxed">
            {errorMessage}
          </p>
          <button
            onClick={handleVerify}
            className="text-xs font-medium text-slate-900 dark:text-white underline underline-offset-2 pt-1 inline-block"
          >
            Try again
          </button>
        </div>
      )}

      {/* Verification Result (Sections 9–19 & 30 Hierarchy) */}
      {result && !isLoading && (
        <div className="space-y-6 animate-fadeIn">
          {/* 1. Verdict & Confidence Card */}
          <VerdictCard
            verdict={result.overall_verdict}
            claimText={result.raw_input}
            multipleClaims={result.claims.length > 1}
          />

          {/* 2. Atomic Claims Analyzed (for multi-claim articles) */}
          <ClaimBreakdown
            claims={result.claims}
            selectedClaimId={selectedClaimId}
            onSelectClaim={setSelectedClaimId}
          />

          {/* 3. Conflicting Evidence Section */}
          <ConflictingAlert 
            evidences={result.all_evidence}
            verdictLabel={result.overall_verdict.verdict}
          />

          {/* 4. Temporal & Statistical Context */}
          <ContextAnalysis
            claimText={result.raw_input}
            isOutdated={result.overall_verdict.is_outdated}
            isMisleading={result.overall_verdict.is_misleading}
            events={result.timeline_events}
          />

          {/* 5. Evidence Summary & Source List */}
          <EvidenceSection
            evidences={result.all_evidence}
            selectedClaimId={selectedClaimId}
          />

          {/* 6. Verification Trace & Technical Analysis */}
          <VerificationTrace result={result} />
        </div>
      )}
    </div>
  );
};
