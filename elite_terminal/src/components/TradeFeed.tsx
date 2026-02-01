import React from 'react';
import { ShieldAlert } from 'lucide-react';

interface Trade {
    id: string;
    price: number;
    size: number;
    side: 'BUY' | 'SELL';
    timestamp: number;
    is_mev: boolean;
    latency_ms: number;
}

interface TradeFeedProps {
    trades: Trade[];
}

export function TradeFeed({ trades }: TradeFeedProps) {
    return (
        <div className="w-full h-full flex flex-col bg-white">
            <div className="enterprise-header flex justify-between items-center">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Global Order Stream</span>
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-teal-500" />
                    <span className="text-[9px] font-bold text-teal-600 uppercase">Live_Feed</span>
                </div>
            </div>

            <div className="flex-grow overflow-y-auto">
                <table className="enterprise-table">
                    <thead className="sticky top-0 bg-white z-10">
                        <tr className="border-b border-gray-100">
                            <th className="p-2">Time</th>
                            <th className="p-2 text-right">Price</th>
                            <th className="p-2 text-right">Size</th>
                            <th className="p-2 text-center">Latency</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                        {trades.map((trade, i) => (
                            <tr key={trade.id || i}>
                                <td className="p-2 text-[10px] text-gray-400 tabular-nums">
                                    {new Date(trade.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </td>
                                <td className={`p-2 text-right text-[11px] font-bold tabular-nums ${trade.side === 'BUY' ? 'text-[#0D9488]' : 'text-[#E11D48]'}`}>
                                    {trade.price.toLocaleString(undefined, { minimumFractionDigits: 1 })}
                                </td>
                                <td className="p-2 text-right text-[10px] text-gray-600 tabular-nums font-bold">
                                    {trade.size.toFixed(4)}
                                </td>
                                <td className="p-2 text-center">
                                    <div className="flex items-center justify-center gap-1.5">
                                        <span className={`text-[10px] font-bold tabular-nums ${trade.latency_ms < 50 ? 'text-gray-400' : 'text-amber-600'}`}>
                                            {trade.latency_ms}ms
                                        </span>
                                        {trade.is_mev && (
                                            <ShieldAlert size={10} className="text-[#1E40AF]" strokeWidth={3} />
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
