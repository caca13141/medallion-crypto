import React, { useState, useMemo } from 'react';

interface OrderBookHeatmapProps {
    data: any;
}

export function OrderBookHeatmap({ data }: OrderBookHeatmapProps) {
    const [hoveredLevel, setHoveredLevel] = useState<{ price: number; volume: number; side: 'bid' | 'ask'; cumulative: number } | null>(null);
    const [depthRange, setDepthRange] = useState<number>(25);

    // MUST call hooks before any conditional returns (Rules of Hooks)
    const metrics = useMemo(() => {
        if (!data || !data.bids || !data.asks || !Array.isArray(data.bids) || !Array.isArray(data.asks)) {
            return null;
        }

        const bids = data.bids.slice(0, depthRange);
        const asks = data.asks.slice(0, depthRange);

        if (bids.length === 0 || asks.length === 0) return null;

        const bestBid = bids[0][0];
        const bestAsk = asks[0][0];
        const midPrice = (bestBid + bestAsk) / 2;
        const spread = bestAsk - bestBid;
        const spreadBps = (spread / midPrice) * 10000;

        const totalBidVol = bids.reduce((sum: number, [_, vol]: number[]) => sum + vol, 0);
        const totalAskVol = asks.reduce((sum: number, [_, vol]: number[]) => sum + vol, 0);
        const imbalance = totalBidVol / (totalBidVol + totalAskVol);

        const maxVol = Math.max(...bids.map((b: any) => b[1]), ...asks.map((a: any) => a[1]));

        return { bestBid, bestAsk, midPrice, spread, spreadBps, totalBidVol, totalAskVol, imbalance, maxVol };
    }, [data, depthRange]);

    // NOW we can do conditional returns
    if (!metrics) {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center bg-white text-gray-400">
                <span className="text-[10px] font-bold uppercase tracking-widest">Awaiting Depth Stream</span>
            </div>
        );
    }

    const bidsWithCumulative = data.bids.slice(0, depthRange).map((bid: number[], idx: number) => ({
        price: bid[0],
        volume: bid[1],
        cumulative: data.bids.slice(0, idx + 1).reduce((sum: number, b: number[]) => sum + b[1], 0)
    }));

    const asksWithCumulative = data.asks.slice(0, depthRange).map((ask: number[], idx: number) => ({
        price: ask[0],
        volume: ask[1],
        cumulative: data.asks.slice(0, idx + 1).reduce((sum: number, a: number[]) => sum + a[1], 0)
    }));

    return (
        <div className="w-full h-full bg-white flex flex-col">
            {/* Metrics Header */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100 shrink-0">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-bold text-gray-400 uppercase">Spread</span>
                        <span className="text-[11px] font-mono font-bold text-gray-700">
                            ${metrics.spread.toFixed(2)}
                            <span className="text-gray-400 ml-1">({metrics.spreadBps.toFixed(1)}bp)</span>
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-bold text-gray-400 uppercase">Imbalance</span>
                        <div className="w-20 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                            <div
                                className={`h-full ${metrics.imbalance > 0.5 ? 'bg-teal-500' : 'bg-rose-500'}`}
                                style={{ width: `${metrics.imbalance * 100}%` }}
                            />
                        </div>
                        <span className="text-[10px] font-mono font-bold text-gray-600">
                            {(metrics.imbalance * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <span className="text-[9px] font-bold text-gray-400 uppercase">Levels</span>
                    {[10, 25, 50].map(range => (
                        <button
                            key={range}
                            onClick={() => setDepthRange(range)}
                            className={`text-[10px] font-bold px-2 py-0.5 rounded ${depthRange === range
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-50 text-gray-500 hover:bg-gray-100'
                                }`}
                        >
                            {range}
                        </button>
                    ))}
                </div>
            </div>

            {/* Depth Visualization */}
            <div className="flex-1 flex relative overflow-hidden min-h-0">
                {/* Bids (Left Side) - Scrollable */}
                <div className="flex-1 flex flex-col-reverse gap-px p-2 overflow-y-auto ">
                    {[...bidsWithCumulative].reverse().map((bid, idx) => {
                        const widthPercent = (bid.volume / metrics.maxVol) * 100;
                        const isHovered = hoveredLevel?.price === bid.price && hoveredLevel?.side === 'bid';

                        return (
                            <div
                                key={idx}
                                className="relative group cursor-pointer"
                                onMouseEnter={() => setHoveredLevel({ price: bid.price, volume: bid.volume, side: 'bid', cumulative: bid.cumulative })}
                                onMouseLeave={() => setHoveredLevel(null)}
                            >
                                <div className="flex items-center justify-end h-4">
                                    <div
                                        className={`h-full rounded-sm transition-all duration-500 ease-in-out ${isHovered ? 'bg-teal-600 opacity-100' : 'bg-teal-500 opacity-70 hover:opacity-90'
                                            }`}
                                        style={{
                                            width: `${widthPercent}%`,
                                            transition: 'width 0.5s ease-in-out, background-color 0.2s'
                                        }}
                                    />
                                </div>
                                {isHovered && (
                                    <div className="absolute right-full mr-2 top-0 bg-gray-900 text-white px-2 py-1 rounded text-[9px] font-mono whitespace-nowrap z-10 shadow-lg">
                                        <div className="font-bold text-teal-400">${bid.price.toLocaleString()}</div>
                                        <div>Vol: {bid.volume.toFixed(2)}</div>
                                        <div>Cum: {bid.cumulative.toFixed(2)}</div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Mid Price Line */}
                <div className="w-px bg-gradient-to-b from-transparent via-yellow-500 to-transparent shrink-0" />

                {/* Asks (Right Side) - Scrollable */}
                <div className="flex-1 flex flex-col-reverse gap-px p-2 overflow-y-auto">
                    {[...asksWithCumulative].reverse().map((ask, idx) => {
                        const widthPercent = (ask.volume / metrics.maxVol) * 100;
                        const isHovered = hoveredLevel?.price === ask.price && hoveredLevel?.side === 'ask';

                        return (
                            <div
                                key={idx}
                                className="relative group cursor-pointer"
                                onMouseEnter={() => setHoveredLevel({ price: ask.price, volume: ask.volume, side: 'ask', cumulative: ask.cumulative })}
                                onMouseLeave={() => setHoveredLevel(null)}
                            >
                                <div className="flex items-center h-4">
                                    <div
                                        className={`h-full rounded-sm transition-all duration-500 ease-in-out ${isHovered ? 'bg-rose-600 opacity-100' : 'bg-rose-500 opacity-70 hover:opacity-90'
                                            }`}
                                        style={{
                                            width: `${widthPercent}%`,
                                            transition: 'width 0.5s ease-in-out, background-color 0.2s'
                                        }}
                                    />
                                </div>
                                {isHovered && (
                                    <div className="absolute left-full ml-2 top-0 bg-gray-900 text-white px-2 py-1 rounded text-[9px] font-mono whitespace-nowrap z-10 shadow-lg">
                                        <div className="font-bold text-rose-400">${ask.price.toLocaleString()}</div>
                                        <div>Vol: {ask.volume.toFixed(2)}</div>
                                        <div>Cum: {ask.cumulative.toFixed(2)}</div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-4 py-2 border-t border-gray-100 shrink-0">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-teal-500 rounded-sm" />
                    <span className="text-[9px] font-bold text-gray-500 uppercase">Bids</span>
                    <span className="text-[10px] font-mono font-bold text-gray-700">{metrics.totalBidVol.toFixed(2)}</span>
                </div>
                <div className="text-[11px] font-mono font-bold text-gray-400">
                    Mid: ${metrics.midPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-bold text-gray-700">{metrics.totalAskVol.toFixed(2)}</span>
                    <span className="text-[9px] font-bold text-gray-500 uppercase">Asks</span>
                    <div className="w-2 h-2 bg-rose-500 rounded-sm" />
                </div>
            </div>
        </div>
    );
}
