export function LoopScoreWaterfall() {
    const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

    return (
        <div className="flex flex-col gap-1 h-full justify-center">
            {timeframes.map((tf, i) => {
                const score = Math.random(); // Mock score 0-1
                const isHigh = score > 0.7;
                return (
                    <div key={tf} className="flex items-center gap-2">
                        <div className="w-8 text-[10px] font-mono text-gray-500 text-right">{tf}</div>
                        <div className="flex-1 h-2 bg-gray-900 rounded-full overflow-hidden relative">
                            <div
                                className="absolute top-0 left-0 h-full bg-cyan-500 transition-all duration-1000"
                                style={{ width: `${score * 100}%`, opacity: isHigh ? 1 : 0.5 }}
                            />
                            {isHigh && <div className="absolute top-0 right-0 h-full w-full bg-white/20 animate-pulse" />}
                        </div>
                        <div className="w-8 text-[10px] font-mono text-right" style={{ color: isHigh ? '#00ffff' : '#666' }}>
                            {score.toFixed(2)}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
