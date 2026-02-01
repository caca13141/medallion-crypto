export function DarkPoolHeatmap() {
    return (
        <div className="w-full h-full bg-black relative overflow-hidden rounded border border-white/5">
            {/* Grid Pattern */}
            <div className="absolute inset-0 opacity-20"
                style={{ backgroundImage: 'radial-gradient(#333 1px, transparent 1px)', backgroundSize: '10px 10px' }}
            />

            {/* Heatmap Blobs */}
            <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-purple-900/40 rounded-full blur-xl animate-pulse" />
            <div className="absolute bottom-1/3 right-1/4 w-24 h-24 bg-blue-900/40 rounded-full blur-xl animate-pulse delay-700" />

            {/* Text Overlay */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                    <div className="text-xs font-mono text-purple-400 tracking-widest mb-1">DARK POOL DEPTH</div>
                    <div className="text-2xl font-black text-white/10">HIDDEN LIQUIDITY</div>
                </div>
            </div>
        </div>
    );
}
