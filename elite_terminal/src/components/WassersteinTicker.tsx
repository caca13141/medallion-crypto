import { useEffect, useState } from 'react';
import { clsx } from 'clsx';

export function WassersteinTicker({ value = 0.0420 }: { value?: number }) {
    const [prev, setPrev] = useState(value);
    const [flash, setFlash] = useState<'up' | 'down' | null>(null);

    useEffect(() => {
        if (value > prev) setFlash('up');
        else if (value < prev) setFlash('down');

        const timer = setTimeout(() => setFlash(null), 500);
        setPrev(value);
        return () => clearTimeout(timer);
    }, [value]);

    return (
        <div className="flex flex-col items-center justify-center p-2 bg-black/40 rounded border border-white/5">
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Wasserstein Dist</div>
            <div className={clsx(
                "text-3xl font-mono font-black tracking-tighter transition-colors duration-300",
                flash === 'up' ? "text-red-500" : flash === 'down' ? "text-cyan-500" : "text-white"
            )}>
                {value.toFixed(6)}
            </div>
            <div className="flex gap-1 mt-1">
                {Array.from({ length: 10 }).map((_, i) => (
                    <div
                        key={i}
                        className={clsx(
                            "w-1 h-1 rounded-full",
                            i < value * 100 ? "bg-cyan-500" : "bg-gray-800"
                        )}
                    />
                ))}
            </div>
        </div>
    );
}
