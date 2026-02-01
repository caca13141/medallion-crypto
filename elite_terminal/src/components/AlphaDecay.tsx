import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface AlphaDecayProps {
    data: { horizon: string; ic: number }[];
}

export function AlphaDecay({ data }: AlphaDecayProps) {
    if (!data || data.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                Awaiting Alpha Vector
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col bg-white">
            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="0" stroke="#F3F4F6" vertical={false} />
                        <XAxis
                            dataKey="horizon"
                            fontSize={9}
                            tickLine={false}
                            axisLine={false}
                            tick={{ fill: '#9CA3AF', fontWeight: 'bold' }}
                        />
                        <YAxis
                            fontSize={9}
                            tickLine={false}
                            axisLine={false}
                            tick={{ fill: '#9CA3AF' }}
                            domain={[0, 0.3]}
                        />
                        <Tooltip
                            cursor={{ fill: '#F9FAFB' }}
                            contentStyle={{
                                backgroundColor: '#FFFFFF',
                                border: '1px solid #E0E0E0',
                                borderRadius: '0',
                                fontSize: '10px',
                                boxShadow: 'none'
                            }}
                            labelStyle={{ fontWeight: 'bold', color: '#111111', marginBottom: '4px' }}
                        />
                        <Bar dataKey="ic" radius={[1, 1, 0, 0]} animationDuration={0}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.ic > 0.1 ? '#1E40AF' : '#60A5FA'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <div className="mt-2 flex justify-between items-center border-t border-gray-50 pt-2 px-1">
                <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest">Persistence_Horizon:</span>
                <span className="text-[11px] font-bold text-[#1E40AF]">Active</span>
            </div>
        </div>
    );
}
