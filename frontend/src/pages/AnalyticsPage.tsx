import React, { useState, useEffect } from 'react';
import { BarChart3, Database, Cpu, TrendingUp, RefreshCw } from 'lucide-react';
import { getAnalytics } from '../services/api';

export const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    getAnalytics()
      .then((data) => {
        setAnalytics(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching analytics:', err);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-20 text-center text-slate-400 flex items-center justify-center gap-2">
        <RefreshCw className="w-5 h-5 animate-spin text-teal-400" />
        <span>Loading system metrics and evaluation models...</span>
      </div>
    );
  }

  const modelMetrics = analytics?.ml_model_evaluation || {};
  const verdictDist = analytics?.verdict_distribution || {};

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          System Analytics & Empirical Evaluation
        </h2>
        <p className="text-sm sm:text-base text-slate-400">
          Live statistics from the SQLite verification audit trail and reproducible model benchmark metrics.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-semibold uppercase text-slate-400">Total Ingested Articles</span>
            <Database className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-3xl font-extrabold text-white">
            {analytics?.total_verifications || 0}
          </div>
          <span className="text-[11px] text-slate-500 font-mono mt-1 block">Persisted in SQLite DB</span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-semibold uppercase text-slate-400">Trained Baselines</span>
            <Cpu className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-3xl font-extrabold text-white">
            {Object.keys(modelMetrics).length || 4}
          </div>
          <span className="text-[11px] text-slate-500 font-mono mt-1 block">NB, LogReg, Linear SVM, RF</span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-semibold uppercase text-slate-400">Retrieval Mode</span>
            <TrendingUp className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-3xl font-extrabold text-teal-400">
            {analytics?.retrieval_mode?.toUpperCase() || 'HYBRID'}
          </div>
          <span className="text-[11px] text-slate-500 font-mono mt-1 block">Live Web + Benchmark Cache</span>
        </div>
      </div>

      {/* Empirical Model Comparison Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-6">
        <div className="border-b border-slate-800 pb-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-teal-400" />
            Machine Learning Baseline Benchmark Comparison
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Empirically measured on hold-out test split of the TruthLens benchmark corpus.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase">
                <th className="py-3 px-4">Classifier Architecture</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 px-4">F1-Score</th>
                <th className="py-3 px-4">Macro-F1</th>
                <th className="py-3 px-4">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {Object.entries(modelMetrics).map(([key, m]: [string, any]) => (
                <tr key={key} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4 font-bold text-white">
                    {key.replace('_', ' ').toUpperCase()}
                  </td>
                  <td className="py-3.5 px-4 text-teal-300">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4">{(m.precision * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4">{(m.recall * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4 font-bold text-teal-400">{(m.f1 * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4">{(m.macro_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4 text-emerald-400 font-bold">{m.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Real-time Verdict Distribution */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-teal-400" />
          Live Verdict History Distribution
        </h3>
        <p className="text-xs text-slate-400">
          Distribution of determinations stored in SQLite across user sessions:
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 pt-2">
          {Object.entries(verdictDist).map(([verdict, count]: [string, any]) => (
            <div key={verdict} className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 text-center">
              <span className="block text-xs font-mono font-bold text-slate-300 mb-1">{verdict}</span>
              <span className="text-2xl font-black text-teal-400">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
