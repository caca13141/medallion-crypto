import React, { useState, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import {
    ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, ReferenceLine
} from 'recharts';
import * as THREE from 'three';

const VolatilitySurface = () => {
    const meshRef = useRef<THREE.Mesh>(null);
    const geometry = useMemo(() => {
        const geom = new THREE.PlaneGeometry(50, 50, 60, 60);
        const pos = geom.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            pos.setZ(i, Math.sin(pos.getX(i) / 5) * Math.cos(pos.getY(i) / 5) * 3);
        }
        return geom;
    }, []);

    useFrame((state) => {
        if (!meshRef.current) return;
        const time = state.clock.getElapsedTime();
        const pos = meshRef.current.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            const z = Math.sin(pos.getX(i) / 5 + time) * Math.cos(pos.getY(i) / 5 + time) * 2;
            pos.setZ(i, z);
        }
        pos.needsUpdate = true;
    });

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2.5, 0, 0]}>
            <meshStandardMaterial color="#1E40AF" wireframe transparent opacity={0.3} side={THREE.DoubleSide} />
        </mesh>
    );
};

const CurvatureSurface = ({ paths, roughness = 0.5 }: { paths: any[], roughness?: number }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const geometry = useMemo(() => {
        const g = new THREE.PlaneGeometry(10, 10, 32, 32);
        const pos = g.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            const x = pos.getX(i);
            const y = pos.getY(i);
            const dist = Math.sqrt(x * x + y * y);
            const warp = Math.sin(dist * 2 - Date.now() * 0.001) * 0.2;
            pos.setZ(i, warp);
        }
        g.computeVertexNormals();
        return g;
    }, [paths]);

    useFrame((state) => {
        if (!meshRef.current) return;
        const pos = meshRef.current.geometry.attributes.position;
        const time = state.clock.getElapsedTime();
        // Scale warping intensity by roughness (higher roughness = more volatile/distorted surface)
        const intensity = 0.3 * (1 + Math.abs(0.5 - roughness) * 3);

        for (let i = 0; i < pos.count; i++) {
            const x = pos.getX(i);
            const y = pos.getY(i);
            const z = Math.sin(x * 1.5 + time) * Math.cos(y * 1.5 + time) * intensity;
            pos.setZ(i, z);
        }
        pos.needsUpdate = true;
    });

    return (
        <mesh ref={meshRef} rotation={[-Math.PI / 3, 0, 0]} geometry={geometry}>
            <meshStandardMaterial
                wireframe
                color="#3B82F6"
                emissive="#0D9488"
                emissiveIntensity={0.2}
                transparent
                opacity={0.15}
                roughness={0}
                metalness={1}
            />
        </mesh>
    );
};

const PersistenceCloud = () => {
    const pointsRef = useRef<THREE.Points>(null);
    const geometry = useMemo(() => {
        const count = 4000;
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            pos[i * 3] = (Math.random() - 0.5) * 60;
            pos[i * 3 + 1] = Math.random() * 20;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 40;
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        return geom;
    }, []);

    return (
        <points ref={pointsRef} geometry={geometry}>
            <pointsMaterial size={0.15} color="#1E40AF" transparent opacity={0.4} />
        </points>
    );
};

const MarketCurvature = ({ volatility }: { volatility: number }) => {
    return (
        <group rotation={[-Math.PI / 4, 0, 0]}>
            <gridHelper args={[60, 20, '#E0E0E0', '#F3F4F6']} />
            <mesh position={[0, 2, 0]}>
                <torusKnotGeometry args={[10, 0.5, 128, 16]} />
                <meshStandardMaterial color="#0D9488" wireframe />
            </mesh>
        </group>
    );
};

interface QuantLabProps {
    marketData: any;
    forecastData: any;
    ghostData?: any[];
    manifoldData?: any[];
    mcSimulation?: {
        paths: number[][];
        last_price: number;
        timestamp: number;
        regime?: string;
        greeks?: {
            delta: number;
            gamma: number;
            var_95: number;
            cvar_95: number;
            avar_95: number;
        };
        roughness?: number;
    } | null;
    intelligence_audit?: {
        current_dominance: number;
        loss_curve: number[];
        expert_landscape: number[];
        learning_rate_stability: number;
        convergent: boolean;
        status: string;
    } | null;
}

export const QuantLab: React.FC<QuantLabProps> = ({ marketData, forecastData, mcSimulation, intelligence_audit }) => {
    const [activeModel, setActiveModel] = useState<'surface' | 'cloud' | 'curvature'>('surface');

    // Greeks Heatmap Component
    const GreeksGrid = () => (
        <div className="grid grid-cols-4 gap-2 mb-4 p-3 bg-slate-50 border border-slate-100 rounded-sm">
            <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">DELTA_ITM</span>
                <span className="text-[14px] font-black text-slate-800 tracking-tighter">
                    {(mcSimulation?.greeks?.delta || 0).toFixed(4)}
                </span>
                <div className="h-1 bg-slate-200 mt-1">
                    <div className="h-full bg-blue-500" style={{ width: `${(mcSimulation?.greeks?.delta || 0) * 100}%` }} />
                </div>
            </div>
            <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">GAMMA_RISK</span>
                <span className="text-[14px] font-black text-slate-800 tracking-tighter">
                    {(mcSimulation?.greeks?.gamma || 0).toFixed(4)}
                </span>
                <div className="h-1 bg-slate-200 mt-1">
                    <div className="h-full bg-rose-500" style={{ width: `${Math.min((mcSimulation?.greeks?.gamma || 0) * 50, 100)}%` }} />
                </div>
            </div>
            <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">ADV_VaR</span>
                <span className="text-[14px] font-black text-rose-600 tracking-tighter">
                    {(mcSimulation?.greeks?.avar_95 || 0).toFixed(2)}%
                </span>
                <div className="h-1 bg-slate-200 mt-1">
                    <div className="h-full bg-rose-600" style={{ width: `${Math.min(Math.abs(mcSimulation?.greeks?.avar_95 || 0) * 10, 100)}%` }} />
                </div>
            </div>
            <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">ROUGHNESS</span>
                <span className="text-[14px] font-black text-teal-600 tracking-tighter">
                    {(mcSimulation?.roughness || 0.5).toFixed(4)}
                </span>
                <div className="h-1 bg-slate-200 mt-1">
                    <div className="h-full bg-teal-500" style={{ width: `${(mcSimulation?.roughness || 0.5) * 100}%` }} />
                </div>
            </div>
        </div>
    );

    // Intelligence Audit Overlay (V11 Power Tracking)
    const IntelligenceAuditLayer = () => {
        const audit = intelligence_audit || {
            current_dominance: 15.4,
            loss_curve: Array(20).fill(0).map(() => Math.random() * 0.5),
            expert_landscape: [0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.05, 0.05],
            status: "CONVERGING"
        };

        return (
            <div className="absolute bottom-16 right-6 z-20 w-56 flex flex-col gap-2 pointer-events-none">
                <div className="bg-white/90 backdrop-blur-md border border-slate-100 p-3 shadow-xl rounded-sm">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">MOE_INTELLIGENCE_AUDIT</span>
                        <span className={`text-[7px] font-bold px-1 rounded-full ${audit.status === 'CONVERGING' ? 'bg-teal-100 text-teal-600' : 'bg-blue-100 text-blue-600'}`}>
                            {audit.status}
                        </span>
                    </div>

                    <div className="mb-3">
                        <div className="flex justify-between items-end mb-1">
                            <span className="text-[9px] font-black text-slate-800 uppercase">Topological_Dominance</span>
                            <span className="text-[14px] font-black text-blue-600 tracking-tighter">{(audit.current_dominance || 0).toFixed(2)}</span>
                        </div>
                        <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-600" style={{ width: `${Math.min(audit.current_dominance * 5, 100)}%` }} />
                        </div>
                    </div>

                    <div className="mb-2">
                        <span className="text-[7px] font-bold text-slate-400 uppercase tracking-widest mb-1 block">Expert_Utilization_Map</span>
                        <div className="flex items-end gap-[1px] h-8">
                            {(audit.expert_landscape || []).map((w: number, i: number) => (
                                <div key={i} className="flex-1 bg-teal-500/20 border-t border-teal-500/50" style={{ height: `${w * 100}%` }} />
                            ))}
                        </div>
                    </div>

                    <div className="pt-2 border-t border-slate-50">
                        <span className="text-[7px] font-bold text-slate-400 uppercase tracking-widest mb-1 block">Entropy_Convergence</span>
                        <div className="flex items-end gap-[1px] h-6">
                            {(audit.loss_curve || []).slice(-20).map((l: number, i: number) => (
                                <div key={i} className="flex-1 bg-rose-500/20" style={{ height: `${(1.0 - l) * 100}%` }} />
                            ))}
                        </div>
                    </div>
                </div>
                <div className="text-right">
                    <span className="text-[7px] font-black text-slate-300 uppercase italic">Power is the ability to see the manifold first.</span>
                </div>
            </div>
        );
    };

    const processedPaths = useMemo(() => {
        if (!mcSimulation || !mcSimulation.paths || mcSimulation.paths.length === 0) return [];

        const { paths } = mcSimulation;
        if (!paths[0] || paths[0].length === 0) return [];

        const numSteps = paths[0].length;
        const numPaths = paths.length;

        const chartData = [];
        for (let t = 0; t < numSteps; t++) {
            const stepPrices = paths.map(p => p[t]).filter(v => isFinite(v)).sort((a, b) => a - b);
            if (stepPrices.length === 0) continue;

            chartData.push({
                step: t,
                min: stepPrices[0],
                p05: stepPrices[Math.floor(numPaths * 0.05)] || stepPrices[0],
                p10: stepPrices[Math.floor(numPaths * 0.10)] || stepPrices[0],
                p20: stepPrices[Math.floor(numPaths * 0.20)] || stepPrices[0],
                p30: stepPrices[Math.floor(numPaths * 0.30)] || stepPrices[0],
                p40: stepPrices[Math.floor(numPaths * 0.40)] || stepPrices[0],
                p50: stepPrices[Math.floor(numPaths * 0.50)] || stepPrices[0],
                p60: stepPrices[Math.floor(numPaths * 0.60)] || stepPrices[0],
                p70: stepPrices[Math.floor(numPaths * 0.70)] || stepPrices[0],
                p80: stepPrices[Math.floor(numPaths * 0.80)] || stepPrices[0],
                p90: stepPrices[Math.floor(numPaths * 0.90)] || stepPrices[stepPrices.length - 1],
                p95: stepPrices[Math.floor(numPaths * 0.95)] || stepPrices[stepPrices.length - 1],
                max: stepPrices[stepPrices.length - 1],
                path1: paths[0] ? paths[0][t] : null,
                path2: paths[5] ? paths[5][t] : null,
                path3: paths[10] ? paths[10][t] : null,
            });
        }
        return chartData;
    }, [mcSimulation]);

    const stats = useMemo(() => {
        if (processedPaths.length === 0) return null;
        const lastStep = processedPaths[processedPaths.length - 1];
        const startPrice = mcSimulation?.last_price || 100000;

        const var95 = ((lastStep.p05 - startPrice) / startPrice) * 100;
        const cvar95 = 1.2 * var95;

        return {
            var95: var95.toFixed(2),
            cvar95: cvar95.toFixed(2),
            upside: (((lastStep.p95 - startPrice) / startPrice) * 100).toFixed(2),
            drift: (((lastStep.p50 - startPrice) / startPrice) * 100).toFixed(2)
        };
    }, [processedPaths, mcSimulation]);

    return (
        <div className="grid grid-cols-12 gap-4 h-full">
            <div className="col-span-8 enterprise-card flex flex-col p-6">
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <div className="text-[10px] font-black text-blue-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                            TERMINAL_VELOCITY_V8_FRONT_INOV
                            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        </div>
                        <h2 className="text-[14px] font-black text-[#111111] uppercase tracking-tight">Probabilistic Price Manifold</h2>
                    </div>

                    <div className="flex gap-4">
                        <div className="flex flex-col items-end">
                            <span className="text-[9px] font-bold text-gray-400 uppercase">HMM_REGIME</span>
                            <span className="text-[12px] font-black text-blue-600 tracking-tighter uppercase">{mcSimulation?.regime || "STABLE"}</span>
                        </div>
                        <div className="flex flex-col items-end border-l border-gray-100 pl-4 group relative cursor-help">
                            <span className="text-[9px] font-bold text-gray-400 uppercase">VaR (95%)</span>
                            <span className="text-[12px] font-bold text-[#E11D48] tracking-tighter">{(mcSimulation?.greeks?.var_95 || 0).toFixed(2)}%</span>
                        </div>
                        <div className="flex flex-col items-end border-l border-gray-100 pl-4 group relative cursor-help">
                            <span className="text-[9px] font-bold text-gray-400 uppercase">CVaR (95%)</span>
                            <span className="text-[12px] font-bold text-[#E11D48] tracking-tighter">{(mcSimulation?.greeks?.cvar_95 || 0).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>

                <GreeksGrid />

                <div className="flex-1 min-h-0 relative bg-white rounded-sm overflow-hidden border border-gray-100">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={processedPaths} margin={{ top: 20, right: 30, left: 10, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                            <XAxis dataKey="step" hide />
                            <YAxis
                                domain={['auto', 'auto']}
                                stroke="#94A3B8"
                                fontSize={9}
                                tickLine={false}
                                axisLine={false}
                                orientation="left"
                                mirror
                            />
                            <Tooltip
                                contentStyle={{ border: '1px solid #E2E8F0', borderRadius: '4px', fontSize: '10px', backgroundColor: 'rgba(255, 255, 255, 0.95)', color: '#0F172A', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                                itemStyle={{ padding: '0px' }}
                                labelStyle={{ display: 'none' }}
                                formatter={(value: any) => [`$${value.toLocaleString()}`, '']}
                            />

                            <defs>
                                <linearGradient id="fanGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.15} />
                                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                                </linearGradient>
                            </defs>

                            <ReferenceLine y={mcSimulation?.last_price} stroke="#CBD5E1" strokeDasharray="4 4" label={{ position: 'right', value: 'SPOT', fill: '#64748B', fontSize: 10, fontWeight: '800' }} />

                            {/* EVOLVING PROBABILITY FAN (High-Density Bright Mode) */}
                            <Area type="monotone" dataKey="p95" stroke="none" fill="#3B82F6" fillOpacity={0.05} animationDuration={800} />
                            <Area type="monotone" dataKey="p90" stroke="none" fill="#3B82F6" fillOpacity={0.08} animationDuration={800} />
                            <Area type="monotone" dataKey="p80" stroke="none" fill="#3B82F6" fillOpacity={0.12} animationDuration={800} />
                            <Area type="monotone" dataKey="p70" stroke="none" fill="#3B82F6" fillOpacity={0.15} animationDuration={800} />
                            <Area type="monotone" dataKey="p60" stroke="#3B82F6" strokeWidth={0.5} strokeOpacity={0.2} fill="#3B82F6" fillOpacity={0.2} animationDuration={800} />

                            {/* Individual Sample Paths (The 'Aura' - Adjusted for White) */}
                            <Line type="monotone" dataKey="path1" stroke="#0D9488" strokeWidth={1} dot={false} strokeOpacity={0.25} isAnimationActive={false} />
                            <Line type="monotone" dataKey="path2" stroke="#0D9488" strokeWidth={0.5} dot={false} strokeOpacity={0.15} isAnimationActive={false} />
                            <Line type="monotone" dataKey="path3" stroke="#0D9488" strokeWidth={0.3} dot={false} strokeOpacity={0.1} isAnimationActive={false} />

                            {/* Institutional Median (Equilibrium) */}
                            <Line
                                type="monotone"
                                dataKey="p50"
                                stroke="#1E293B"
                                strokeWidth={2}
                                dot={false}
                                animationDuration={1000}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>

                    {/* LVP: LIQUIDITY VOLUMETRIC PROFILE (Vertical Density Overlay) */}
                    <div className="absolute right-0 top-0 bottom-0 w-24 flex flex-col justify-between py-10 pointer-events-none pr-1">
                        {useMemo(() => {
                            if (processedPaths.length === 0) return [];
                            const lastStep = mcSimulation?.paths.map(p => p[p.length - 1]).filter(v => isFinite(v)) || [];
                            if (lastStep.length === 0) return [];

                            const minPrice = Math.min(...lastStep);
                            const maxPrice = Math.max(...lastStep);
                            const buckets = 40;
                            const bucketSize = (maxPrice - minPrice) / buckets;
                            const counts = new Array(buckets).fill(0);

                            lastStep.forEach(p => {
                                const b = Math.min(Math.floor((p - minPrice) / bucketSize), buckets - 1);
                                counts[b]++;
                            });

                            const maxC = Math.max(...counts);
                            return counts.reverse().map((c, i) => (
                                <div key={i} className="flex justify-end items-center h-[2px]">
                                    <div
                                        className={`transition-all duration-1000 ${c > maxC * 0.7 ? 'bg-blue-600/60' : 'bg-slate-300/20'}`}
                                        style={{ width: `${(c / maxC) * 100}%`, height: '100%' }}
                                    />
                                </div>
                            ));
                        }, [processedPaths, mcSimulation])}
                    </div>

                    {/* HUD OVERLAYS (Bright Mode) */}
                    <div className="absolute right-4 top-4 pointer-events-none text-[8px] font-black tracking-widest text-slate-400 uppercase space-y-1">
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-[1px] bg-blue-200" />
                            <span>P95_CONFIDENCE_SHELL</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-[1px] bg-teal-200" />
                            <span>STOCHASTIC_SAMPLE_AURA</span>
                        </div>
                    </div>

                    <div className="absolute right-0 top-0 bottom-0 w-16 flex flex-col justify-between py-6 px-1 text-[7px] font-black text-slate-300 uppercase tracking-tighter select-none pointer-events-none bg-gradient-to-l from-slate-50 to-transparent">
                        <span className="text-blue-500/50">Bull_Extension</span>
                        <span>Upper_Cluster</span>
                        <span className="text-slate-600/50">Equilibrium_Core</span>
                        <span>Lower_Cluster</span>
                        <span className="text-rose-500/50">Tail_Risk_Zone</span>
                    </div>
                </div>
            </div>

            <div className="col-span-4 enterprise-card flex flex-col relative overflow-hidden">
                <div className="absolute top-6 left-6 z-10 w-64 group relative cursor-help">
                    <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">MANIFOLD_SCAN</div>
                    <div className="flex items-center gap-2">
                        <h2 className="text-[14px] font-bold text-[#111111] uppercase tracking-tight">{activeModel}</h2>
                        <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
                    </div>
                    <div className="absolute top-full left-0 mt-2 w-full bg-white border border-gray-100 p-2 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                        <p className="text-[8px] leading-tight text-gray-600 font-medium">
                            <span className="font-bold text-black uppercase">Manifold Dynamics:</span> This 3D mesh shows "Price Stress".
                            <br /><br />
                            - **Flat Mesh**: Market in equilibrium.
                            - **Dips/Warps**: High probability of price cascades (liquidity holes).
                        </p>
                    </div>
                </div>

                <div className="absolute bottom-6 left-6 z-10 flex gap-1">
                    {['surface', 'cloud', 'curvature'].map(m => (
                        <button
                            key={m}
                            onClick={() => setActiveModel(m as any)}
                            className={`px-3 py-1 text-[9px] font-bold uppercase border ${activeModel === m ? 'bg-[#111111] text-white border-[#111111]' : 'text-gray-400 border-gray-100 bg-white/80'}`}
                        >
                            {m}
                        </button>
                    ))}
                </div>

                <div className="flex-1 bg-white relative">
                    <IntelligenceAuditLayer />
                    <Canvas>
                        <PerspectiveCamera makeDefault position={[0, 5, 10]} />
                        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                        <ambientLight intensity={1} />
                        <pointLight position={[10, 10, 10]} intensity={1} />

                        {activeModel === 'surface' && <CurvatureSurface paths={processedPaths} roughness={mcSimulation?.roughness} />}
                        {activeModel === 'cloud' && <PersistenceCloud />}
                        {activeModel === 'curvature' && <CurvatureSurface paths={processedPaths} roughness={mcSimulation?.roughness} />}
                    </Canvas>
                </div>

                <div className="h-32 border-t border-gray-100 p-4 bg-white">
                    <div className="text-[9px] font-black text-gray-400 uppercase mb-3 tracking-widest flex justify-between items-center">
                        <span>Terminal_PDF (Risk Heatmap)</span>
                        <span className="text-teal-600/50">Ensemble_Count: 1000</span>
                    </div>
                    <div className="flex items-end gap-[2px] h-12">
                        {useMemo(() => {
                            if (processedPaths.length === 0) return [];
                            const lastStep = mcSimulation?.paths.map(p => p[p.length - 1]).filter(v => isFinite(v)) || [];
                            if (lastStep.length === 0) return [];

                            const minPrice = Math.min(...lastStep);
                            const maxPrice = Math.max(...lastStep);
                            const bucketSize = (maxPrice - minPrice) / 40;
                            const buckets = Array(40).fill(0);

                            lastStep.forEach(p => {
                                const bIdx = Math.min(Math.floor((p - minPrice) / bucketSize), 39);
                                buckets[bIdx]++;
                            });

                            const maxCount = Math.max(...buckets);
                            return buckets.map((count, i) => (
                                <div
                                    key={i}
                                    className={`flex-1 transition-all duration-700 ${count > maxCount * 0.5 ? 'bg-blue-600/30' : 'bg-blue-400/10'}`}
                                    style={{ height: `${(count / maxCount) * 100}%`, borderTop: count > 0 ? '1px solid rgba(37, 99, 235, 0.4)' : 'none' }}
                                />
                            ));
                        }, [processedPaths, mcSimulation])}
                    </div>
                </div>
            </div>
        </div>
    );
};
