import React from 'react';
import { ShieldCheck, Sun, Moon } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
  historyCount?: number;
}

export const Navbar: React.FC<NavbarProps> = ({ 
  activeTab, 
  setActiveTab, 
  theme, 
  setTheme,
  historyCount = 0
}) => {
  const navItems = [
    { id: 'verify', label: 'Verify' },
    { id: 'sources', label: 'Sources' },
    { id: 'history', label: 'History', badge: historyCount > 0 ? historyCount : undefined },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-white/85 dark:bg-[#050505]/85 border-b border-slate-200 dark:border-neutral-800/80 transition-colors">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Brand Wordmark */}
        <div 
          onClick={() => setActiveTab('verify')}
          className="flex items-center gap-2.5 cursor-pointer group select-none"
        >
          <div className="w-7 h-7 rounded-lg bg-slate-900 text-white dark:bg-white dark:text-black flex items-center justify-center font-bold text-sm shadow-sm">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
              TruthLens
            </span>
            <span className="hidden sm:inline-block text-[11px] font-medium text-slate-400 dark:text-neutral-400">
              Evidence Verification
            </span>
          </div>
        </div>

        {/* Minimal Navigation Links */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`relative px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  isActive
                    ? 'text-slate-900 dark:text-white bg-slate-100 dark:bg-neutral-800/80 shadow-xs'
                    : 'text-slate-500 hover:text-slate-900 dark:text-neutral-400 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-neutral-900/50'
                }`}
              >
                <span>{item.label}</span>
                {item.badge !== undefined && (
                  <span className="ml-1.5 px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-300">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Status Indicator & Theme Switcher */}
        <div className="flex items-center gap-3">
          {/* Live evidence status */}
          <div 
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 dark:bg-[#0e0e11] border border-slate-200 dark:border-neutral-800 text-[11px] text-slate-600 dark:text-neutral-300 select-none"
            title="External retrieval mode: Live Web with corroborating primary evidence archive"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            <span className="font-medium text-[11px]">Live evidence</span>
          </div>

          {/* Theme Control */}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-1.5 rounded-md text-slate-500 hover:text-slate-900 dark:text-neutral-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-neutral-800/70 transition-colors"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <Sun className="w-4 h-4 text-neutral-400 hover:text-amber-300 transition-colors" />
            ) : (
              <Moon className="w-4 h-4 text-slate-600 hover:text-slate-900 transition-colors" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
