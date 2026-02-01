import { useMemo } from 'react';

interface WhaleTrade {
    price: number;
    size: number;
    value: number;
    side: string;
    timestamp: number;
}

interface WhaleFlowProps {
    data?: WhaleTrade[];
}

export function WhaleFlowHeatmap({ data }: WhaleFlowProps) {
    const buckets = useMemo(() => {
        if (!data || data.length === 0) {
            // Fallback mock data if no real data yet
            return Array.from({ length: 60 }, (_, i) => ({
                net: Math.sin(i / 5) * 500000 + (Math.random() - 0.5) * 200000,
                timestamp: Date.now() - (59 - i) * 60000
            }));
        }

        // Aggregate into 1-minute buckets
        const now = Date.now();
        const buckets = new Map<number, number>();

        // Initialize last 60 minutes
        for (let i = 0; i < 60; i++) {
            const bucketTime = Math.floor((now - i * 60000) / 60000) * 60000;
            buckets.set(bucketTime, 0);
        }

        data.forEach(t => {
            const bucketTime = Math.floor(t.timestamp / 60000) * 60000;
            if (buckets.has(bucketTime)) {
                const val = t.side === 'BUY' ? t.value : -t.value;
                buckets.set(bucketTime, (buckets.get(bucketTime) || 0) + val);
            }
        });

        return Array.from(buckets.entries())
            .sort((a, b) => a[0] - b[0])
            .map(([ts, net]) => ({ timestamp: ts, net }));

    }, [data]);

    return (
        <div className="w-full h-full flex flex-col">
            <div className="flex-1 flex items-end gap-[1px]">
                {buckets.map((b, i) => {
                    const intensity = Math.min(Math.abs(b.net) / 1000000, 1); // Cap at $1M net flow
                    const height = Math.max(10, intensity * 100);
                    const color = b.net > 0 ? `rgba(0, 255, 255, ${0.3 + intensity * 0.7})` : `rgba(255, 0, 85, ${0.3 + intensity * 0.7})`;

                    return (
                        <div
                            key={b.timestamp}
                            className="flex-1 rounded-t-[1px] transition-all duration-500 hover:opacity-100 opacity-80 relative group"
                            style={{
                                height: `${height}%`,
                                backgroundColor: color,
                                boxShadow: intensity > 0.8 ? `0 0 10px ${b.net > 0 ? '#00ffff' : '#ff0055'}` : 'none'
                            }}
                        >
                            {/* Tooltip */}
                            <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-black/90 border border-white/20 p-1 text-[9px] font-mono whitespace-nowrap hidden group-hover:block z-50 rounded">
                                {new Date(b.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                <br />
                                <span className={b.net > 0 ? "text-cyan-400" : "text-red-400"}>
                                    ${(b.net / 1000).toFixed(0)}k
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
            <div className="h-4 flex justify-between text-[8px] font-mono text-gray-600 mt-1 border-t border-white/5 pt-1">
                <span>-60m</span>
                <span>NET WHALE FLOW (1m)</span>
                <span>NOW</span>
            </div>
        </div>
    );
}
