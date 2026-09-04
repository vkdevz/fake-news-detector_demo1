import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { VerifyPage } from './pages/VerifyPage';
import { SourcesPage } from './pages/SourcesPage';
import { HistoryPage, VerificationHistoryItem } from './pages/HistoryPage';
import { VerificationResponse } from './types/verification';
import { API_BASE } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('verify');
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('truthlens-theme');
    return (saved === 'light' || saved === 'dark') ? saved : 'dark';
  });
  const [history, setHistory] = useState<VerificationHistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem('truthlens-session-history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [selectedHistoryResult, setSelectedHistoryResult] = useState<VerificationResponse | null>(null);

  // Sync theme with document element
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('truthlens-theme', theme);
  }, [theme]);

  // Persist history
  useEffect(() => {
    try {
      localStorage.setItem('truthlens-session-history', JSON.stringify(history.slice(0, 30)));
    } catch {
      // Storage unavailable or full
    }
  }, [history]);

  // Keep-alive heartbeat: ping backend health endpoint every 10 minutes to prevent Render idle spin-down
  useEffect(() => {
    const keepAliveInterval = setInterval(() => {
      fetch(`${API_BASE}/health`)
        .then((res) => {
          if (res.ok) {
            console.log('🔄 RENDER KEEP-ALIVE HEARTBEAT: Client ping [HTTP 200]');
          }
        })
        .catch(() => {});
    }, 10 * 60 * 1000);

    return () => clearInterval(keepAliveInterval);
  }, []);

  const handleVerificationComplete = (result: VerificationResponse) => {
    const newItem: VerificationHistoryItem = {
      id: result.request_id || String(Date.now()),
      claim: result.raw_input,
      verdict: result.overall_verdict.verdict,
      confidence: result.overall_verdict.confidence,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      result
    };

    setHistory((prev) => [newItem, ...prev.filter(h => h.claim !== newItem.claim)].slice(0, 30));
  };

  const handleSelectHistoryItem = (item: VerificationHistoryItem) => {
    setSelectedHistoryResult(item.result);
    setActiveTab('verify');
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem('truthlens-session-history');
    } catch {}
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#050505] text-slate-900 dark:text-zinc-100 selection:bg-teal-500/20 selection:text-teal-700 dark:selection:text-teal-300 transition-colors">
      {/* Top Navigation */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        theme={theme}
        setTheme={setTheme}
        historyCount={history.length}
      />

      {/* Main Content Area */}
      <main className="flex-1 pb-16">
        {activeTab === 'verify' && (
          <VerifyPage 
            onVerificationComplete={handleVerificationComplete}
            externalResult={selectedHistoryResult}
          />
        )}
        {activeTab === 'sources' && <SourcesPage />}
        {activeTab === 'history' && (
          <HistoryPage 
            history={history}
            onSelectHistoryItem={handleSelectHistoryItem}
            onClearHistory={handleClearHistory}
          />
        )}
      </main>

      {/* Professional Minimal Footer */}
      <footer className="border-t border-slate-200/80 dark:border-neutral-800/80 bg-white dark:bg-[#050505] py-6 transition-colors">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-800 dark:text-zinc-200">TruthLens</span>
            <span>—</span>
            <span>Evidence-grounded verification for claims, statements, and sources.</span>
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            Multi-source corroboration & context analysis
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
