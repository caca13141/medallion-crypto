import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface ForecastData {
    tti: number;
    confidence: number;
    curve: number[];
    regime: string;
    components: {
        topo: number;
        cde: number;
        hawkes: number;
    };
}

interface ForecastPanelProps {
    data: ForecastData | null;
}

export const ForecastPanel: React.FC<ForecastPanelProps> = ({ data }) => {
    if (!data) {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center bg-gray-50 text-gray-400">
                <span className="text-[10px] font-bold uppercase tracking-widest">Ensemble Warmup</span>
            </div>
        );
    }

    const chartData = (data.curve || []).map((val, i) => ({
        step: i,
        tti: val
    }));

    return (
        <div className="w-full h-full flex flex-col bg-white">
            <div className="flex justify-between items-start mb-4">
                <div className="flex gap-6">
                    <div>
                        <div className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mb-1">REGIME_STATE</div>
                        <div className={`text-[12px] font-bold ${data.regime === 'High Volatility' ? 'text-[#E11D48]' : 'text-[#1E40AF]'}`}>
                            {data.regime.toUpperCase()}
                        </div>
                    </div>
                    <div>
                        <div className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mb-1">CONFIDENCE</div>
                        <div className="text-[12px] font-bold text-[#111111]">
                            {(data.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>

                <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] text-gray-400 font-bold uppercase">TOPO</span>
                        <span className="text-[10px] font-mono font-bold text-gray-900">{data.components?.topo?.toFixed(2) ?? '0.00'}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] text-gray-400 font-bold uppercase">CDE</span>
                        <span className="text-[10px] font-mono font-bold text-gray-900">{data.components?.cde?.toFixed(2) ?? '0.00'}</span>
                    </div>
                </div>
            </div>

            <div className="flex-1 w-full min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="0" stroke="#F3F4F6" vertical={false} />
                        <XAxis
                            dataKey="step"
                            hide
                        />
                        <YAxis
                            stroke="#D1D5DB"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#ffffff',
                                border: '1px solid #E0E0E0',
                                borderRadius: '0px',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                                fontSize: '10px',
                                padding: '4px 8px'
                            }}
                            labelStyle={{ display: 'none' }}
                        />
                        <Line
                            type="monotone"
                            dataKey="tti"
                            stroke="#1E40AF"
                            strokeWidth={1.5}
                            dot={false}
                            animationDuration={0}
                        />
                        <ReferenceLine y={data.tti} stroke="#0D9488" strokeWidth={1} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-2 flex justify-between">
                <span className="text-[9px] font-bold text-gray-400 uppercase">Current</span>
                <span className="text-[9px] font-bold text-gray-400 uppercase">Projected (48H)</span>
            </div>
        </div>
    );
};
