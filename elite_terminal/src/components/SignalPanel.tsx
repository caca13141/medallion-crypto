import React from 'react';
import { Activity, Zap, Info } from 'lucide-react';

interface SignalPanelProps {
    data: any;
}

export function SignalPanel({ data }: SignalPanelProps) {
    if (!data) {
        return (
            <div className="p-4 text-gray-400 text-[10px] font-bold uppercase tracking-widest">
                Awaiting Data
            </div>
        );
    }

    const { tti, neural_cde, ppo_action, cascade_prob, entropy, stationarity } = data;

    const MetricRow = ({ label, value, subtext, color = 'text-[#111111]' }: any) => (
        <div className="flex justify-between items-center py-2.5 border-b border-gray-50 last:border-0">
            <div className="flex flex-col">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest leading-none mb-1">
                    {label}
                </span>
                <span className="text-[10px] text-gray-400 font-medium">{subtext}</span>
            </div>
            <span className={`text-[13px] font-bold tabular-nums ${color}`}>{value}</span>
        </div>
    );

    return (
        <div className="flex flex-col h-full bg-white">
            <div className="mb-4">
                <div className="flex items-center gap-2 mb-4 bg-blue-50/50 p-2 border border-blue-100/50">
                    <Activity size={14} className="text-[#1E40AF]" />
                    <span className="text-[11px] font-bold text-[#1E40AF] uppercase tracking-tighter">
                        Predictive Ensemble
                    </span>
                </div>

                <MetricRow
                    label="Topo_TTI"
                    value={tti?.toFixed(2) || "0.00"}
                    subtext="Order Flow Imbalance"
                    color={tti > 7 ? 'text-[#0D9488]' : tti < 3 ? 'text-[#E11D48]' : 'text-[#111111]'}
                />
                <MetricRow
                    label="Neural_CDE"
                    value={`${((neural_cde?.edge || 0) * 100).toFixed(1)}%`}
                    subtext="Latent Path Integral"
                />
                <MetricRow
                    label="PPO_Policy"
                    value={ppo_action?.side || "N/A"}
                    subtext={`Size: ${(ppo_action?.size || 0).toFixed(1)}x`}
                    color={ppo_action?.side === 'BUY' ? 'text-[#0D9488]' : 'text-[#E11D48]'}
                />
                <MetricRow
                    label="Flow Toxicity"
                    value={`${((cascade_prob || 0) * 100).toFixed(1)}%`}
                    subtext="Microstructure Risk"
                    color={cascade_prob > 0.7 ? 'text-[#E11D48]' : 'text-[#111111]'}
                />
            </div>

            <div className="mt-auto border-t-2 border-gray-100 pt-4">
                <div className="flex items-center gap-2 mb-4 bg-gray-50 p-2 border border-gray-200">
                    <Zap size={14} className="text-gray-600" />
                    <span className="text-[11px] font-bold text-gray-600 uppercase tracking-tighter">
                        Market Physics
                    </span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <div className="flex flex-col p-2 bg-[#F9FAFB] border border-gray-100">
                        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mb-1">
                            Entropy
                        </span>
                        <span className="text-[12px] font-bold text-[#111111]">
                            {entropy?.toFixed(3) || "0.742"}
                        </span>
                        <div className="mt-1 h-1 w-full bg-gray-200 overflow-hidden">
                            <div className="h-full bg-gray-400" style={{ width: `${(entropy || 0.7) * 100}%` }} />
                        </div>
                    </div>
                    <div className="flex flex-col p-2 bg-[#F9FAFB] border border-gray-100">
                        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mb-1">
                            Stationarity
                        </span>
                        <span className={`text-[12px] font-bold ${stationarity < -3 ? 'text-[#0D9488]' : 'text-[#E11D48]'}`}>
                            {stationarity?.toFixed(2) || "-3.24"}
                        </span>
                        <div className="mt-1 h-1 w-full bg-gray-200 overflow-hidden">
                            <div
                                className={`h-full ${stationarity < -3 ? 'bg-[#0D9488]' : 'bg-[#E11D48]'}`}
                                style={{ width: `${Math.min(100, Math.abs(stationarity || -3) * 25)}%` }}
                            />
                        </div>
                    </div>
                </div>

                <div className="mt-3 p-2 bg-amber-50 border border-amber-100 flex items-start gap-2">
                    <Info size={10} className="text-amber-600 shrink-0 mt-0.5" />
                    <p className="text-[8px] font-bold text-amber-900 uppercase leading-tight">
                        {stationarity > -2.5
                            ? "Non-Ergodic Regime Detected. Mean Reversion invalid."
                            : "Stationarity Confirmed. Gaussian dynamics expected."}
                    </p>
                </div>
            </div>
        </div>
    );
}
