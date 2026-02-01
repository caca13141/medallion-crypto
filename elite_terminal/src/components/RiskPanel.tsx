import React, { useEffect, useState } from 'react';
import { ShieldCheck, Zap, AlertCircle, Droplets } from 'lucide-react';

interface RiskPanelProps {
    data: any;
    vpin?: number;
}

export function RiskPanel({ data, vpin = 0.15 }: RiskPanelProps) {
    if (!data) {
        return (
            <div className="text-gray-400 text-[9px] font-bold uppercase tracking-widest">
                Safety: Init
            </div>
        );
    }

    const { kill_switch_active, cooldown_end, tti_threshold, daily_pnl_pct, hrp_weights, alpha_decay } = data;
    const [timeLeft, setTimeLeft] = useState<string>("");

    useEffect(() => {
        if (!cooldown_end) return;
        const interval = setInterval(() => {
            const diff = new Date(cooldown_end).getTime() - Date.now();
            if (diff <= 0) {
                setTimeLeft("00:00");
            } else {
                const m = Math.floor(diff / 60000);
                const s = Math.floor((diff % 60000) / 1000);
                setTimeLeft(`${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [cooldown_end]);

    return (
        <div className="flex flex-col gap-4">
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-4 border-r border-gray-200 pr-6">
                    <div className="flex flex-col text-right">
                        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest leading-none mb-1">PnL_Daily</span>
                        <span className={`text-[12px] font-bold tabular-nums ${daily_pnl_pct >= 0 ? 'text-[#0D9488]' : 'text-[#E11D48]'}`}>
                            {daily_pnl_pct > 0 ? '+' : ''}{daily_pnl_pct.toFixed(2)}%
                        </span>
                    </div>
                    <div className="flex flex-col text-right">
                        <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest leading-none mb-1">T_Hold</span>
                        <span className="text-[12px] font-bold text-[#111111] tabular-nums">{(tti_threshold ?? 8.0).toFixed(1)}</span>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <div className={`pill h-7 px-3 flex items-center gap-2 ${kill_switch_active ? 'pill-error' : 'pill-success'}`}>
                        {kill_switch_active ? <AlertCircle size={10} strokeWidth={3} /> : <ShieldCheck size={10} strokeWidth={3} />}
                        <span className="tracking-tight uppercase text-[10px] font-bold">
                            {kill_switch_active ? `LOCKED: ${timeLeft}` : 'SECURE'}
                        </span>
                    </div>

                    <button className="h-7 px-4 bg-[#111111] hover:bg-black text-[#FFFFFF] text-[10px] font-bold uppercase rounded-[1px] flex items-center gap-2 transition-none">
                        <Zap size={10} fill="currentColor" />
                        Flatten
                    </button>
                </div>
            </div>

            <div className="border-t border-gray-100 pt-3">
                <div className="flex justify-between items-center mb-1.5">
                    <div className="flex items-center gap-1.5">
                        <Droplets size={10} className={vpin > 0.7 ? 'text-[#E11D48]' : 'text-[#1E40AF]'} />
                        <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">Order_Flow_Toxicity (VPIN)</span>
                    </div>
                    <span className={`text-[10px] font-bold tabular-nums ${vpin > 0.7 ? 'text-[#E11D48]' : 'text-[#1E40AF]'}`}>
                        {(vpin * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-500 ${vpin > 0.7 ? 'bg-[#E11D48]' : 'bg-[#1E40AF]'}`}
                        style={{ width: `${vpin * 100}%` }}
                    />
                </div>
            </div>

            {/* Institutional: HRP Weights & Alpha Decay */}
            <div className="grid grid-cols-2 gap-4 mt-2">
                <div className="flex flex-col gap-2">
                    <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">HRP_ALLOCATION</span>
                    <div className="flex items-end gap-[2px] h-10 border-b border-slate-100 pb-1">
                        {(hrp_weights || [0.3, 0.2, 0.4, 0.1]).map((w: number, i: number) => (
                            <div key={i} className="flex-1 bg-slate-800" style={{ height: `${w * 100}%` }} />
                        ))}
                    </div>
                </div>
                <div className="flex flex-col gap-2">
                    <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">ALPHA_DECAY</span>
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${(alpha_decay || 0) < 0.2 ? 'bg-teal-500' : 'bg-rose-500'} animate-pulse`} />
                        <span className="text-[10px] font-bold tabular-nums text-slate-700">
                            {((alpha_decay || 0.04) * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div className="h-1 bg-slate-100">
                        <div className="h-full bg-blue-500" style={{ width: `${(1 - (alpha_decay || 0.04)) * 100}%` }} />
                    </div>
                </div>
            </div>
        </div>
    );
}
