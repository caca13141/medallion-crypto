interface LiquidationLadderProps {
    currentPrice?: number;
    liquidations?: any[];
}

export function LiquidationLadder({ currentPrice = 43000, liquidations = [] }: LiquidationLadderProps) {
    // Generate estimated liquidation levels relative to current price
    const levels = [
        { price: currentPrice * 1.02, vol: 12.5, type: 'short' },
        { price: currentPrice * 1.01, vol: 45.2, type: 'short' },
        { price: currentPrice * 1.005, vol: 8.1, type: 'short' },
        { price: currentPrice * 0.995, vol: 22.4, type: 'long' },
        { price: currentPrice * 0.99, vol: 156.0, type: 'long' },
    ];

    return (
        <div className="flex flex-col gap-1 h-full overflow-hidden relative">
            <div className="absolute top-1/2 left-0 w-full h-[1px] bg-white/20 z-10" /> {/* Current Price Line */}

            {/* Real Liquidation Events Overlay */}
            {liquidations && liquidations.slice(0, 5).map((liq, i) => (
                <div key={i} className="absolute right-0 z-20 animate-ping" style={{
                    top: liq.side === 'SELL' ? '20%' : '80%', // Rough positioning based on side
                    opacity: 0.5
                }}>
                    <div className={clsx("w-2 h-2 rounded-full", liq.side === 'SELL' ? "bg-red-500" : "bg-green-500")} />
                </div>
            ))}

            {levels.map((lvl, i) => (
                <div key={i} className="flex items-center justify-between text-[10px] font-mono hover:bg-white/5 px-1 rounded cursor-crosshair group">
                    <span className={lvl.type === 'short' ? "text-red-400" : "text-green-400"}>
                        ${lvl.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </span>
                    <div className="flex items-center gap-2">
                        <div
                            className={clsx("h-1 rounded-full transition-all group-hover:h-2", lvl.type === 'short' ? "bg-red-500/50" : "bg-green-500/50")}
                            style={{ width: `${Math.min(lvl.vol, 100)}px` }}
                        />
                        <span className="text-gray-600 w-8 text-right">{lvl.vol.toFixed(1)}M</span>
                    </div>
                </div>
            ))}

            {/* Recent Real Liquidations List */}
            <div className="absolute bottom-0 left-0 w-full bg-black/80 p-1 text-[8px] font-mono border-t border-white/10">
                <div className="text-gray-500 mb-0.5">RECENT LIQUIDATIONS</div>
                {liquidations.length === 0 ? (
                    <div className="text-gray-700 italic">No recent events</div>
                ) : (
                    liquidations.slice(0, 3).map((l, i) => (
                        <div key={i} className="flex justify-between">
                            <span className={l.side === 'SELL' ? "text-red-400" : "text-green-400"}>
                                {l.side} ${parseFloat(l.price).toLocaleString()}
                            </span>
                            <span className="text-gray-400">{parseFloat(l.origQty).toFixed(3)} BTC</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

import { clsx } from 'clsx';
