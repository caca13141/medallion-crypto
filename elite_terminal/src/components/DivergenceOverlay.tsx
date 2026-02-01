import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine } from 'recharts';
import { useState, useEffect } from 'react';

interface PredictionPoint {
    time: string;
    actual: number | null;
    predicted: number;
    divergence: number;
}

interface DivergenceOverlayProps {
    data?: PredictionPoint[];
}

export function DivergenceOverlay({ data }: DivergenceOverlayProps) {
    const [chartData, setChartData] = useState<PredictionPoint[]>([]);

    useEffect(() => {
        if (data) {
            setChartData(data);
            return;
        }

        // Generate mock 48h prediction data
        const mockData: PredictionPoint[] = [];
        const now = Date.now();
        const basePrice = 43200;

        for (let i = -24; i <= 24; i++) {
            const timestamp = now + i * 60 * 60 * 1000; // Hourly
            const time = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            // Actual price (only for past data)
            const actual = i <= 0 ? basePrice + Math.sin(i / 5) * 1000 + Math.random() * 200 : null;

            // Predicted price (entire 48h window)
            const predicted = basePrice + Math.sin(i / 5) * 1000 + (i > 0 ? 300 : 0) + Math.random() * 150;

            // Divergence (only where we have actual data)
            const divergence = actual !== null ? Math.abs(actual - predicted) : 0;

            mockData.push({ time, actual, predicted, divergence });
        }

        setChartData(mockData);
    }, [data]);

    // Calculate max divergence for color scaling
    const maxDivergence = Math.max(...chartData.map(d => d.divergence));

    return (
        <div className="h-full w-full relative">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                    <defs>
                        {/* Gradient for divergence zones */}
                        <linearGradient id="divergenceGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ff0055" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#ff0055" stopOpacity={0} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />

                    <XAxis
                        dataKey="time"
                        stroke="#444"
                        tick={{ fill: '#666', fontSize: 10 }}
                        interval="preserveStartEnd"
                    />

                    <YAxis
                        stroke="#444"
                        tick={{ fill: '#666', fontSize: 10 }}
                        domain={['auto', 'auto']}
                    />

                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#0a0a0a',
                            border: '1px solid #333',
                            borderRadius: '4px',
                            fontSize: '11px',
                        }}
                        labelStyle={{ color: '#00ffff' }}
                        itemStyle={{ color: '#fff' }}
                        formatter={(value: any, name: string) => {
                            if (name === 'divergence') return [`${value.toFixed(2)}`, 'Divergence'];
                            return [`$${value?.toFixed(2)}`, name === 'actual' ? 'Actual' : 'Predicted'];
                        }}
                    />

                    {/* Vertical line separating past/future */}
                    <ReferenceLine
                        x={chartData.find(d => d.actual === null)?.time}
                        stroke="#00ffff"
                        strokeDasharray="5 5"
                        strokeWidth={2}
                        label={{
                            value: 'NOW',
                            position: 'top',
                            fill: '#00ffff',
                            fontSize: 10,
                            fontWeight: 'bold'
                        }}
                    />

                    {/* Divergence area (red zone) */}
                    <Area
                        type="monotone"
                        dataKey="divergence"
                        stroke="none"
                        fillOpacity={1}
                        fill="url(#divergenceGradient)"
                        isAnimationActive={false}
                    />

                    {/* Actual price line (dashed white for past data) */}
                    <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#ffffff"
                        strokeWidth={2}
                        dot={false}
                        strokeDasharray="5 5"
                        connectNulls={false}
                    />

                    {/* Predicted price line (solid cyan) */}
                    <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#00ffff"
                        strokeWidth={3}
                        dot={false}
                        isAnimationActive={true}
                        animationDuration={1000}
                    />
                </LineChart>
            </ResponsiveContainer>

            {/* Legend */}
            <div className="absolute top-2 right-2 text-[10px] font-mono space-y-1 bg-black/50 p-2 rounded border border-white/10 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                    <div className="w-4 h-0.5 bg-white" style={{ borderTop: '2px dashed white' }} />
                    <span className="text-white">Actual</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-0.5 bg-cyan-500" />
                    <span className="text-cyan-500">Predicted</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-2 bg-red-500/30" />
                    <span className="text-red-500">Divergence</span>
                </div>
            </div>

            {/* Divergence Alert */}
            {maxDivergence > 500 && (
                <div className="absolute bottom-2 left-2 bg-red-900/50 border border-red-500/50 px-3 py-1 rounded text-xs font-mono text-red-400 backdrop-blur-sm animate-pulse">
                    ⚠ HIGH DIVERGENCE: ${maxDivergence.toFixed(0)}
                </div>
            )}
        </div>
    );
}
