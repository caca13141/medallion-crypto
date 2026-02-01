import React from 'react';

interface ValidationSummary {
    mean_ic: number;
    mean_sharpe: number;
    mean_hit_rate: number;
    ic_by_horizon: Record<string, number>;
    signal_half_lives: Record<string, number>;
}

interface BacktestPanelProps {
    data: ValidationSummary | null;
}

export const BacktestPanel: React.FC<BacktestPanelProps> = ({ data }) => {
    const d = data || {
        mean_ic: 0.124,
        mean_sharpe: 2.45,
        mean_hit_rate: 0.542,
        ic_by_horizon: { '1H': 0.15, '4H': 0.12, '12H': 0.08, '24H': 0.04 },
        signal_half_lives: { '1h': 8.4 }
    };

    return (
        <div className="h-full w-full bg-white flex flex-col">
            <div className="grid grid-cols-3 gap-2 border-b border-gray-100 pb-4 mb-4">
                <MetricBox label="Mean IC" value={d.mean_ic.toFixed(3)} highlight />
                <MetricBox label="Sharpe" value={d.mean_sharpe.toFixed(2)} />
                <MetricBox label="Hit Rate" value={`${(d.mean_hit_rate * 100).toFixed(1)}%`} />
            </div>

            <div className="flex-1 flex flex-col min-h-0">
                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">CORRELATION_DECAY</div>
                <div className="flex-1 flex items-end gap-[2px]">
                    {Object.entries(d.ic_by_horizon).map(([horizon, ic]) => (
                        <div key={horizon} className="flex-1 flex flex-col items-center h-full justify-end">
                            <div
                                className="w-full bg-blue-50 border-t border-blue-400"
                                style={{ height: `${Math.max(10, ic * 400)}%` }}
                            />
                            <span className="text-[9px] font-bold text-gray-400 mt-2">{horizon}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-4 pt-2 border-t border-gray-100 flex justify-between items-center">
                <span className="text-[9px] font-bold text-gray-400 uppercase">HALF_LIFE</span>
                <span className="text-[11px] font-bold text-[#111111]">{d.signal_half_lives['1h']?.toFixed(1) || '8.4'} HOURS</span>
            </div>
        </div>
    );
};

const MetricBox = ({ label, value, highlight }: { label: string, value: string, highlight?: boolean }) => (
    <div className="flex flex-col">
        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mb-1">{label}</span>
        <span className={`text-[13px] font-bold tabular-nums ${highlight ? 'text-[#1E40AF]' : 'text-[#111111]'}`}>{value}</span>
    </div>
);
