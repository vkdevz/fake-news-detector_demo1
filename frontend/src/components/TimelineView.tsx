import React from 'react';
import { History, Calendar } from 'lucide-react';
import { TimelineEvent } from '../types/verification';

interface TimelineViewProps {
  events: TimelineEvent[];
}

export const TimelineView: React.FC<TimelineViewProps> = ({ events }) => {
  if (!events || events.length === 0) {
    return null;
  }

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-3">
        <History className="w-5 h-5 text-purple-400" />
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
          Chronological & Temporal Context Flow
        </h3>
      </div>

      <div className="relative pl-6 space-y-6 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
        {events.map((evt, idx) => {
          const isLatest = idx === events.length - 1;
          return (
            <div key={idx} className="relative group">
              {/* Circle Marker */}
              <div className={`absolute -left-6 top-1 w-4 h-4 rounded-full border-2 ${
                isLatest
                  ? 'bg-purple-500 border-purple-300'
                  : 'bg-slate-900 border-slate-600'
              }`} />

              <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="text-xs font-bold text-slate-200 uppercase font-mono tracking-wider">
                    {evt.title}
                  </span>
                  <span className="flex items-center gap-1 text-[11px] font-mono text-purple-400">
                    <Calendar className="w-3 h-3" />
                    {evt.date}
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {evt.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
