export function LatencyWaterfall() {
    const exchanges = [
        { name: 'BINANCE', lat: 45 },
        { name: 'BYBIT', lat: 32 },
        { name: 'OKX', lat: 120 },
        { name: 'HYPERLIQUID', lat: 15 },
    ];

    return (
        <div className="flex flex-col gap-2">
            {exchanges.map(ex => (
                <div key={ex.name} className="flex items-center gap-2 text-[10px] font-mono">
                    <div className="w-20 text-gray-500">{ex.name}</div>
                    <div className="flex-1 h-1 bg-gray-900 rounded overflow-hidden">
                        <div
                            className={clsx("h-full rounded", ex.lat < 50 ? "bg-green-500" : ex.lat < 100 ? "bg-yellow-500" : "bg-red-500")}
                            style={{ width: `${Math.min(ex.lat, 200) / 2}%` }}
                        />
                    </div>
                    <div className={clsx("w-10 text-right", ex.lat < 50 ? "text-green-500" : "text-gray-500")}>
                        {ex.lat}ms
                    </div>
                </div>
            ))}
        </div>
    );
}

import { clsx } from 'clsx';
