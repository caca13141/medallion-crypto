import React, { useMemo } from 'react';

interface RegimeFingerprintProps {
    dna: number[];
    regime: string;
}

export function RegimeFingerprint({ dna, regime }: RegimeFingerprintProps) {
    if (!dna || dna.length < 5) {
        return (
            <div className="w-full h-full flex items-center justify-center text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                Awaiting DNA Sequence
            </div>
        );
    }

    const labels = ["Trend", "Liquidity", "Vol", "Persistence", "Entropy"];
    const points = useMemo(() => {
        const radius = 55;
        const centerX = 100;
        const centerY = 95;
        return dna.map((val, i) => {
            const angle = (i * 2 * Math.PI) / dna.length - Math.PI / 2;
            const r = (val * 0.8 + 0.2) * radius;
            return {
                x: centerX + r * Math.cos(angle),
                y: centerY + r * Math.sin(angle),
                labelX: centerX + (radius + 25) * Math.cos(angle),
                labelY: centerY + (radius + 15) * Math.sin(angle)
            };
        });
    }, [dna]);

    const polygonPath = points.map(p => `${p.x},${p.y}`).join(' ');

    return (
        <div className="w-full h-full flex flex-col items-center justify-center bg-white p-2">
            <div className="relative w-full h-full max-h-[200px]">
                <svg viewBox="0 0 200 180" className="w-full h-full overflow-visible">
                    {[0.2, 0.4, 0.6, 0.8, 1.0].map(r => (
                        <circle
                            key={r}
                            cx="100" cy="95" r={r * 55}
                            fill="none"
                            stroke="#F3F4F6"
                            strokeWidth="1"
                        />
                    ))}

                    {points.map((p, i) => (
                        <line
                            key={i}
                            x1="100" y1="95" x2={100 + 55 * Math.cos((i * 2 * Math.PI) / dna.length - Math.PI / 2)}
                            y2={95 + 55 * Math.sin((i * 2 * Math.PI) / dna.length - Math.PI / 2)}
                            stroke="#F3F4F6"
                            strokeWidth="1"
                        />
                    ))}

                    <polygon
                        points={polygonPath}
                        fill="#1E40AF"
                        fillOpacity="0.1"
                        stroke="#1E40AF"
                        strokeWidth="1.5"
                    />

                    {points.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r="2.5" fill="#1E40AF" />
                    ))}

                    {points.map((p, i) => (
                        <text
                            key={i}
                            x={p.labelX}
                            y={p.labelY}
                            textAnchor="middle"
                            className="text-[8px] font-bold fill-gray-400 uppercase tracking-tighter"
                        >
                            {labels[i]}
                        </text>
                    ))}
                </svg>
            </div>

            <div className="mt-1 text-center border-t border-gray-50 w-full pt-1">
                <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest">ARCHETYPE: </span>
                <span className="text-[11px] font-bold text-[#111111]">{regime.toUpperCase()}</span>
            </div>
        </div>
    );
}
