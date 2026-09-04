import React from 'react';
import { CheckCircle2, CircleDashed, Circle, AlertCircle, Clock } from 'lucide-react';
import { PipelineStep } from '../types/verification';

interface PipelineStepperProps {
  steps: PipelineStep[];
  isProcessing?: boolean;
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({ steps }) => {
  return (
    <div className="bg-slate-900/70 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Clock className="w-5 h-5 text-teal-400" />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Inspection Pipeline Execution Trace
          </h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          {steps.filter(s => s.status === 'completed').length} / {steps.length} Steps Verified
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {steps.map((step, idx) => {
          const isDone = step.status === 'completed';
          const isInProgress = step.status === 'in_progress';
          const isFailed = step.status === 'failed';

          return (
            <div
              key={step.step_key || idx}
              className={`p-3.5 rounded-xl border transition-all ${
                isDone
                  ? 'bg-slate-950/60 border-teal-500/30 text-slate-200'
                  : isInProgress
                  ? 'bg-teal-950/20 border-teal-500/60 text-teal-200 animate-pulse'
                  : isFailed
                  ? 'bg-rose-950/20 border-rose-500/40 text-rose-300'
                  : 'bg-slate-950/30 border-slate-800/60 text-slate-500'
              }`}
            >
              <div className="flex items-center gap-2.5 mb-1.5">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0" />
                ) : isInProgress ? (
                  <CircleDashed className="w-4 h-4 text-teal-400 animate-spin shrink-0" />
                ) : isFailed ? (
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                ) : (
                  <Circle className="w-4 h-4 text-slate-600 shrink-0" />
                )}
                <span className="text-xs font-medium tracking-tight truncate">
                  {step.label}
                </span>
              </div>

              {step.details && (
                <p className="text-[11px] text-slate-400 leading-relaxed truncate pl-6">
                  {step.details}
                </p>
              )}

              {step.duration_ms !== undefined && (
                <span className="inline-block mt-1 pl-6 text-[10px] font-mono text-slate-500">
                  {step.duration_ms}ms
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
